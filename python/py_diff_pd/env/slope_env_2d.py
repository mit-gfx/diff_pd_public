import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.quad_mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import QuadMesh2d, QuadDeformable, StdRealVector

class SlopeEnv2d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = 4e5
        poissons_ratio = 0.45
        state_force_parameters = options['state_force_parameters']

        # Mesh parameters.
        cell_nums = (4, 4)
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
        dx = 0.05
        origin = (0, 0.001)
        bin_file_name = str(folder / 'mesh.bin')
        generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
        mesh = QuadMesh2d()
        mesh.Initialize(bin_file_name)

        # FEM parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e2
        deformable = QuadDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)

        # External force.
        g = state_force_parameters[:2]
        deformable.AddStateForce('gravity', g)

        # Contact.
        kn, kf, mu = state_force_parameters[2:]
        deformable.AddStateForce('planar_contact', [0, 1, 0, 3, kn, kf, mu])
        # Use these lines to try the penetration-free contact.
        '''
        friction_node_idx = []
        for i in range(node_nums[0]):
            friction_node_idx.append(i * node_nums[1])
        deformable.SetFrictionalBoundary('planar', [0.0, 1.0, 0.0], friction_node_idx)
        '''

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Initial conditions.
        dofs = deformable.dofs()
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
        display_quad_mesh(mesh, xlim=[-0.5, 0.5], ylim=[0.0, 0.5],
            file_name=file_name, show=False)

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)