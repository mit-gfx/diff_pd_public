import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector

class CantileverEnv3d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh. We use 8 for cantilever_3d.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options['refinement'] if 'refinement' in options else 2
        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([5.])
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        cell_nums = (4 * refinement, refinement, refinement)
        origin = ndarray([0, 0, 0])
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
        dx = 0.08 / refinement
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Boundary conditions.
        for j in range(node_nums[1]):
            for k in range(node_nums[2]):
                node_idx = j * node_nums[2] + k
                vx, vy, vz = mesh.py_vertex(node_idx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # Collisions.
        def to_index(i, j, k):
            return i * node_nums[1] * node_nums[2] + j * node_nums[2] + k
        collision_indices = [to_index(cell_nums[0], 0, 0), to_index(cell_nums[0], cell_nums[1], 0)]
        deformable.AddPdEnergy('planar_collision', [1e2, 0.0, 0.0, 1.0, 0.1], collision_indices)
        # Actuation.
        act_indices = []
        for i in range(cell_nums[0]):
            j = 0
            k = 0
            act_indices.append(i * cell_nums[1] * cell_nums[2] + j * cell_nums[2] + k)
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        deformable.AddActuation(actuator_stiffness[0], [1.0, 0.0, 0.0], act_indices)

        # Initial state set by rotating the cuboid kinematically.
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        vertex_num = mesh.NumOfVertices()
        q0 = ndarray(mesh.py_vertices())
        max_theta = np.pi / 2
        for i in range(1, node_nums[0]):
            theta = max_theta * i / (node_nums[0] - 1)
            c, s = np.cos(theta), np.sin(theta)
            R = ndarray([[1, 0, 0],
                [0, c, -s],
                [0, s, c]])
            center = ndarray([i * dx, cell_nums[1] / 2 * dx, cell_nums[2] / 2 * dx]) + origin
            for j in range(node_nums[1]):
                for k in range(node_nums[2]):
                    idx = i * node_nums[1] * node_nums[2] + j * node_nums[2] + k
                    v = ndarray(mesh.py_vertex(idx))
                    q0[3 * idx:3 * idx + 3] = R @ (v - center) + center
        v0 = np.zeros(dofs)
        f_ext = np.random.normal(scale=0.1, size=dofs) * density * (dx ** 3)

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
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.__node_nums = node_nums

        self.__spp = options['spp'] if 'spp' in options else 4

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        i = dof // (self.__node_nums[1] * self.__node_nums[2])
        return i == 0

    def _display_mesh(self, mesh_file, file_name):
        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        render_hex_mesh(mesh, file_name=file_name,
            resolution=(400, 400), sample=self.__spp, transforms=[
                ('t', (-0.16, 0.16, 0.05)),
                ('s', 6)
            ],
            camera_pos=(2, -2.2, 1.4),
            render_voxel_edge=True)

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)