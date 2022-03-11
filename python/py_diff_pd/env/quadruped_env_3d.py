import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector

class QuadrupedEnv3d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.49
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([np.log10(5) + 5])
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])
        # Configure the size of the quadruped:
        # Foot width = 1.
        foot_width_size = 0.1
        leg_z_length = options['leg_z_length'] if 'leg_z_length' in options else 2
        body_x_length = options['body_x_length'] if 'body_x_length' in options else 4
        body_y_length = options['body_y_length'] if 'body_y_length' in options else 4
        body_z_length = options['body_z_length'] if 'body_z_length' in options else 1
        refinement = options['refinement'] if 'refinement' in options else 2
        assert leg_z_length >= 1
        assert body_x_length >= 3
        assert body_y_length >= 3
        assert body_z_length >= 1
        # Refinement defines how many cells does food_width have.
        assert refinement >= 2
        dx = foot_width_size / refinement

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 3e2
        cell_nums = (refinement * body_x_length, refinement * body_y_length, refinement * (leg_z_length + body_z_length))
        origin = ndarray([0, 0, 0])
        node_nums = [n + 1 for n in cell_nums]

        bin_file_name = folder / 'mesh.bin'
        # Topology.
        voxels = np.zeros(cell_nums)
        for i in range(cell_nums[0]):
            for j in range(cell_nums[1]):
                # Body.
                for k in range(refinement * leg_z_length, cell_nums[2]):
                    voxels[i][j][k] = 1
                # Legs.
                for k in range(refinement * leg_z_length):
                    if (i < refinement and j < refinement) or (i < refinement and j >= body_y_length * refinement - refinement) or \
                        (i >= body_x_length * refinement - refinement and j < refinement) or \
                        (i >= body_x_length * refinement - refinement and j >= body_y_length * refinement - refinement):
                        voxels[i][j][k] = 1
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # Collisions.
        vertex_num = mesh.NumOfVertices()
        friction_node_idx = []
        for i in range(vertex_num):
            vx, vy, vz = mesh.py_vertex(i)
            if vz < dx / 2:
                vx_idx = vx / dx
                vy_idx = vy / dx
                if vx_idx < 0.5 or np.abs(vx_idx - refinement) < 0.5 or \
                    np.abs(vx_idx - (body_x_length * refinement - refinement)) < 0.5 or \
                    np.abs(vx_idx - body_x_length * refinement) < 0.5 or \
                    vy_idx < 0.5 or np.abs(vy_idx - refinement) < 0.5 or \
                    np.abs(vy_idx - (body_y_length * refinement - refinement)) < 0.5 or \
                    np.abs(vy_idx - body_y_length * refinement) < 0.5:
                    friction_node_idx.append(i)
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)
        # Actuation: we have 4 legs and each leg has 4 muscles.
        # Convention: F (Front): positive x; R (Rear): negative x;
        #             L (Left): positive y; R (Right): negative y.
        leg_indices = {}
        act_indices = []
        count = 0
        element_num = mesh.NumOfElements()
        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of this cell.
            com = 0
            for vi in v_idx:
                com += ndarray(mesh.py_vertex(int(vi)))
            com /= len(v_idx)
            x_idx, y_idx, z_idx = com / dx
            if z_idx >= leg_z_length * refinement: continue
            # First, determine which leg the voxel is in.
            leg_key = ('F' if x_idx >= body_x_length * refinement * 0.5 else 'R') \
                + ('L' if y_idx >= body_y_length * refinement * 0.5 else 'R')
            # Second, determine which muscle this voxel should be in front (F) or back (B).
            if leg_key[0] == 'F':
                x_idx -= body_x_length * refinement - refinement
            muscle_key = 'F' if x_idx >= 0.5 * refinement else 'B'

            key = leg_key + muscle_key
            if key not in leg_indices:
                leg_indices[key] = [count,]
                count += 1
                act_indices.append(i)
            else:
                leg_indices[key].append(count)
                act_indices.append(i)
                count += 1
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], act_indices)

        # Initial conditions.
        dofs = deformable.dofs()
        print('Quadruped element: {:d}, DoFs: {:d}.'.format(element_num, dofs))
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._actuator_parameters = actuator_parameters
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__node_nums = node_nums
        self.__element_num = element_num
        self._leg_indices = leg_indices
        self._act_indices = act_indices
        self._options = options

        scale = 0.5
        self._spp = options['spp'] if 'spp' in options else 8
        self._camera_pos = (0.25, -1, .25)
        self._camera_lookat = (0.25, 0, 0.1)
        self._color = (0.8, 0.8, 0.3)
        self._scale = scale

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def element_num(self):
        return self.__element_num

    def leg_indices(self):
        return self._leg_indices

    def act_indices(self):
        return self._act_indices

    def _loss_and_grad(self, q, v):
        # Compute the center of mass.
        com = np.mean(q.reshape((-1, 3)), axis=0)
        loss = -com[0]

        grad_q = np.zeros(q.size)
        vertex_num = int(q.size // 3)
        grad_q[::3] = -1 / vertex_num

        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v

    def _stepwise_loss_and_grad(self, q, v, i):
        # Step wise pitch loss
        options = self._options
        leg_z_length = options['leg_z_length']
        body_x_length = options['body_x_length']
        body_y_length = options['body_y_length']
        body_z_length = options['body_z_length']
        refinement = options['refinement']

        #Finds the topmost midpoint on the body face. qb and qf will differ in y initially if refinemnt is odd
        body_slice = ((leg_z_length + body_z_length) * refinement + 1) * 2 * (refinement + 1) + \
            (body_z_length * refinement  + 1) * (body_y_length*refinement - 2*refinement - 1)
        zb_idx = int(np.ceil(body_slice / 2) * 3 + 2)
        zf_idx = int(-np.floor(body_slice / 2) * 3)
        zb = q[zb_idx]
        zf = q[zf_idx]

        grad_q = np.zeros(q.size)
        loss = 0
        if abs(zb - zf) > 0.075:
            loss = 1/200 * (zb - zf) ** 2
            grad_q[zb_idx] = 1/100 * (zb - zf)
            grad_q[zf_idx] = 1/100 * (zf - zb)

        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v
