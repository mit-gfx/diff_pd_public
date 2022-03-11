import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.quad_mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import QuadMesh2d, QuadDeformable, StdRealVector

class HopperEnv2d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options['refinement'] if 'refinement' in options else 2
        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 4e5
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([5., 5.])
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, -9.81])

        # Mesh parameters.
        cell_nums = (2 * refinement, 4 * refinement)
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
        dx = 0.03 / refinement
        origin = (0, dx)
        bin_file_name = str(folder / 'mesh.bin')
        generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
        mesh = QuadMesh2d()
        mesh.Initialize(bin_file_name)

        # FEM parameters.
        youngs_modulus = 4e5
        poissons_ratio = 0.45
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        deformable = QuadDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)

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
            right_muscle_indices.append((2 * refinement - 1) * cell_nums[1] + j)
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        deformable.AddActuation(actuator_stiffness[0], [0.0, 1.0], left_muscle_indices)
        deformable.AddActuation(actuator_stiffness[1], [0.0, 1.0], right_muscle_indices)
        # Collision.
        friction_node_idx = []
        for i in range(node_nums[0]):
            friction_node_idx.append(i * node_nums[1])
        deformable.SetFrictionalBoundary('planar', [0.0, 1.0, 0.0], friction_node_idx)

        # Initial conditions.
        dofs = deformable.dofs()
        # Perturb q0 a bit to avoid singular gradients in SVD.
        q0 = ndarray(mesh.py_vertices()) + np.random.uniform(low=-2 * dx * 0.01, high=2 * dx * 0.01, size=dofs)

        # 5 body lengths per second at 45 deg from the horizontal
        v0_mult = 5 * dx * cell_nums[0]
        v0 = np.ones(dofs) * v0_mult
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._dx = dx
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
        return False

    def _display_mesh(self, mesh_file, file_name):
        mesh = QuadMesh2d()
        mesh.Initialize(mesh_file)
        display_quad_mesh(mesh, xlim=[-0.01, 0.3], ylim=[0, 0.2],
            file_name=file_name, show=False)

    def _loss_and_grad(self, q, v):
        dx = self._dx
        # The L2 norm of the difference in pos and vel state compared to a target
        target_q = np.copy(self._q0)
        target_q[::2] += 9 * dx

        target_v = np.copy(self._v0)

        q_diff = q - target_q
        v_diff = v - target_v
        loss = 0.5 * q_diff.dot(q_diff) + 0.5 * v_diff.dot(v_diff)

        grad_q = q_diff
        grad_v = v_diff
        return loss, grad_q, grad_v
