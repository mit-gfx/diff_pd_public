import time
from pathlib import Path
import os

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_contact_vertex
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.common.project_path import root_path

class CowEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.49
        gait = options['gait'] if 'gait' in options else 'pronk'
        if 'actuator_parameters' in options:
            actuator_parameters = options['actuator_parameters']
        elif gait == 'gallop':
            actuator_parameters = ndarray([np.log10(5) + 5, np.log10(5) + 5])
        else:
            actuator_parameters = ndarray([np.log10(5) + 5,])
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 7e2

        bin_file_name = Path(root_path) / 'asset' / 'mesh' / 'spot.bin'
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        cow_size = 0.1
        # Rescale the mesh.
        mesh.Scale(cow_size)
        tmp_bin_file_name = '.tmp.bin'
        mesh.SaveToFile(tmp_bin_file_name)

        dx =  1/15 * cow_size

        deformable = HexDeformable()
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)
        # Collisions.
        vertex_num = mesh.NumOfVertices()
        friction_node_idx = []
        for i in range(vertex_num):
            vx, vy, vz = mesh.py_vertex(i)
            if vz < dx / 2:
                friction_node_idx.append(i)
        # Uncomment the code below if you would like to display the contact set for a sanity check:

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in friction_node_idx])
        # ax.scatter(v[:, 0], v[:, 1], v[:, 2])
        # plt.show()

        # Friction_node_idx = all vertices on the edge.
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)

        # Actuation: we have 4 legs and each leg has 4 muscles.
        # Convention: F (Front): positive x; R (Rear): negative x;
        #             L (Left): positive y; R (Right): negative y.
        leg_indices = {}
        act_indices = []
        count = 0
        leg_z_length = 4
        body_x_length = 9
        body_y_length = 6
        body_z_length = 4
        foot_x = 2
        foot_y = 2
        x_offset = 1
        y_offset = 1
        element_num = mesh.NumOfElements()
        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of this cell.
            com = 0
            for vi in v_idx:
                com += ndarray(mesh.py_vertex(int(vi)))
            com /= len(v_idx)
            x_idx, y_idx, z_idx = com / dx

            if z_idx >= leg_z_length: continue
            if x_idx >= body_x_length + x_offset or \
                (x_idx >= foot_x + x_offset and x_idx <= body_x_length  + x_offset - foot_x): continue
            if y_idx >= foot_y + y_offset and y_idx <= body_y_length + y_offset - foot_y: continue
            # First, determine which leg the voxel is in.
            #print("i: {}, x: {}, y: {}, z: {}".format(i, x_idx, y_idx, z_idx))
            leg_key = ('F' if x_idx >= body_x_length * 0.5 + x_offset else 'R') \
                + ('L' if y_idx >= body_y_length * 0.5 + y_offset else 'R')
            # Second, determine which muscle this voxel should be in front (F) or back (B).
            if leg_key[0] == 'F':
                x_idx -= body_x_length - foot_x
            muscle_key = ('F' if x_idx >= foot_x/2 + x_offset else 'B') \

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

        spine_indices = {}
        if gait == 'gallop':
            act_indices =[]
            for i in range(element_num):
                v_idx = ndarray(mesh.py_element(i))
                # Obtain the center of this cell.
                com = 0
                for vi in v_idx:
                    com += ndarray(mesh.py_vertex(int(vi)))
                com /= len(v_idx)
                x_idx, y_idx, z_idx = com / dx

                if z_idx < leg_z_length: continue
                if z_idx > leg_z_length + body_z_length: continue
                if x_idx < foot_x + x_offset or \
                    x_idx >= body_x_length  + x_offset - foot_x: continue
                # First, determine whether muscle is on Dorsal side (D) or ventral side (V)
                key = 'D' if z_idx > leg_z_length + body_z_length / 2 else 'V'
                if key not in spine_indices:
                    spine_indices[key] = [count, ]
                    count += 1
                    act_indices.append(i)
                else:
                    spine_indices[key].append(count)
                    act_indices.append(i)
                    count += 1

            deformable.AddActuation(actuator_stiffness[1], [1.0, 0.0, 0.0], act_indices)
        # Initial states.
        dofs = deformable.dofs()
        print('Cow element: {:d}, DoFs: {:d}.'.format(element_num, dofs))
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
        self._leg_indices = leg_indices
        self._act_indices = act_indices
        self._spine_indices = spine_indices
        self.__element_num = mesh.NumOfElements()

        scale = 3
        self._spp = options['spp'] if 'spp' in options else 8
        self._camera_pos = (0.5, -1, 0.3)
        self._camera_lookat = (0.5, 0, 0.15)
        self._color = (0.9, 0.9, 0.3)
        self._scale = scale
        self._resolution = (1600, 900)

    def element_num(self):
        return self.__element_num

    def leg_indices(self):
        return self._leg_indices

    def act_indices(self):
        return self._act_indices

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def _loss_and_grad(self, q, v):
        # Compute the center of mass.
        com = np.mean(q.reshape((-1, 3)), axis=0)
        # Compute loss.
        z_weight = 0.4
        loss = -com[0] - z_weight * com[2]
        # Compute grad.
        grad_q = np.zeros(q.size)
        vertex_num = int(q.size // 3)
        grad_q[::3] = -1 / vertex_num
        grad_q[2::3] = -1 * z_weight / vertex_num
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v
