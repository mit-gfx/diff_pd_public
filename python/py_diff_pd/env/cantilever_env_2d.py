import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.quad_mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import QuadMesh2d, QuadDeformable, StdRealVector

class CantileverEnv2d(EnvBase):
    # Possible options:
    # - refinement: controls the mesh discretization.
    # - youngs_modulus and poissons_ratio: control the material parameters.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options['refinement'] if 'refinement' in options else 2
        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 4e5
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([4., 4.])
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, -9.81])

        # Mesh parameters.
        cell_nums = (2 * refinement, 4 * refinement)
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
        dx = 0.03 / refinement
        origin = (0, 0)
        bin_file_name = str(folder / 'mesh.bin')
        generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
        mesh = QuadMesh2d()
        mesh.Initialize(bin_file_name)

        # FEM parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e4
        deformable = QuadDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)

        # Boundary conditions.
        node_idx = cell_nums[1]
        pivot = ndarray(mesh.py_vertex(node_idx))
        deformable.SetDirichletBoundaryCondition(2 * node_idx, pivot[0])
        deformable.SetDirichletBoundaryCondition(2 * node_idx + 1, pivot[1])

        # External force.
        deformable.AddStateForce('gravity', state_force_parameters)

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Actuation.
        left_muscle_indices = []
        right_muscle_indices = []
        for j in range(cell_nums[1]):
            left_muscle_indices.append(0 * cell_nums[1] + j)
            right_muscle_indices.append(refinement * cell_nums[1] + j)
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        deformable.AddActuation(actuator_stiffness[0], [0.0, 1.0], left_muscle_indices)
        deformable.AddActuation(actuator_stiffness[1], [0.0, 1.0], right_muscle_indices)

        # Collision.
        deformable.AddPdEnergy('planar_collision', [1e3, 0.0, 1.0, -0.036], [
            i * node_nums[1] for i in range(node_nums[0])
        ])

        # Initial conditions.
        dofs = deformable.dofs()
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        R = ndarray([[c, -s],
            [s, c]])
        q0 = ndarray(mesh.py_vertices())
        vertex_num = mesh.NumOfVertices()
        for i in range(vertex_num):
            qi = q0[2 * i:2 * i + 2]
            q0[2 * i:2 * i + 2] = R @ (qi - pivot) + pivot
        v0 = np.zeros(dofs)
        f_ext = np.random.uniform(low=0, high=5, size=dofs) * density * dx * dx

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

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return dof in [2 * (self.__node_nums[1] - 1), 2 * (self.__node_nums[1] - 1) + 1]

    def _display_mesh(self, mesh_file, file_name):
        mesh = QuadMesh2d()
        mesh.Initialize(mesh_file)
        display_quad_mesh(mesh, xlim=[-0.01, 0.15], ylim=[-0.01, 0.2],
            file_name=file_name, show=False)

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)