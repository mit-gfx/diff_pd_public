import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.quad_mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import QuadMesh2d, QuadDeformable, StdRealVector

class RopeEnv2d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh. We use 8 for cantilever_3d.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 4e4
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, -9.81])
        cell_nums = options['cell_nums'] if 'cell_nums' in options else (1024, 8)
        contact_ratio = options['contact_ratio'] if 'contact_ratio' in options else 0.1

        # Mesh parameters.
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
        dx = 0.005 * 1024 / cell_nums[0]  # Ensure the full width is 1024 * 0.005.
        origin = (-cell_nums[0] / 2 * dx, 5.05)
        bin_file_name = str(folder / 'mesh.bin')
        generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
        mesh = QuadMesh2d()
        mesh.Initialize(bin_file_name)

        # FEM parameters.
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

        # Collision.
        friction_node_idx = []
        for i in range(int(node_nums[0] * (0.5 - contact_ratio / 2)), int(node_nums[0] * (0.5 + contact_ratio / 2))):
            friction_node_idx.append(i * node_nums[1])
        deformable.SetFrictionalBoundary('spherical', [0.0, 2.5, 2.5], friction_node_idx)

        # Initial conditions.
        dofs = deformable.dofs()
        # Perturb q0 a bit to avoid singular gradients in SVD.
        q0 = ndarray(mesh.py_vertices()) + np.random.uniform(low=-2 * dx * 0.01, high=2 * dx * 0.01, size=dofs)
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._dx = dx
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)

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
        display_quad_mesh(mesh, xlim=[-8, 8], ylim=[0, 6], file_name=file_name, show=False)

    def _loss_and_grad(self, q, v):
        loss = ndarray(q).ravel().dot(self.__loss_q_grad) + ndarray(v).ravel().dot(self.__loss_v_grad)
        return loss, ndarray(self.__loss_q_grad).copy().ravel(), ndarray(self.__loss_v_grad).copy().ravel()
