from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.quad_mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import QuadMesh2d, QuadDeformable, StdRealVector

class PingpongEnv2d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options['refinement']
        youngs_modulus = options['youngs_modulus']
        poissons_ratio = options['poissons_ratio']

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3

        cell_nums = (refinement, refinement)
        origin = ndarray([0, 0])
        cube_size = 0.1
        dx = cube_size / refinement
        bin_file_name = folder / 'mesh.bin'
        generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
        mesh = QuadMesh2d()
        mesh.Initialize(str(bin_file_name))

        deformable = QuadDeformable()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Collisions.
        friction_node_idx = []
        vertex_num = mesh.NumOfVertices()
        corners = [ndarray([0, 0]),
            ndarray([0, cell_nums[1]]),
            ndarray([cell_nums[0], 0]),
            ndarray([cell_nums[0], cell_nums[1]])]
        for i in range(vertex_num):
            v = ndarray(mesh.py_vertex(i)) / dx
            for c in corners:
                if np.linalg.norm(v - c) < 0.5:
                    friction_node_idx.append(i)
                    break

        # Uncomment the code below if you would like to display the contact set for a sanity check:
        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot()
        v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in friction_node_idx])
        ax.scatter(v[:, 0], v[:, 1])
        plt.show()
        '''

        # Friction_node_idx = all vertices on the edge.
        deformable.SetFrictionalBoundary('planar', [0.0, 1.0, 0.0], friction_node_idx)

        # Initial states.
        dofs = deformable.dofs()
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
        self._stepwise_loss = False
        self._target_q = None

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def set_target_q(self, target_q):
        self._target_q = ndarray(np.copy(target_q)).ravel()

    def _display_mesh(self, mesh_file, file_name):
        mesh = QuadMesh2d()
        mesh.Initialize(mesh_file)
        display_quad_mesh(mesh, xlim=[0, 1], ylim=[0, 1],
            file_name=file_name, show=False)

    def _loss_and_grad(self, q, v):
        assert self._target_q is not None
        q_diff = q - self._target_q
        # Compute loss.
        loss = 0.5 * q_diff.dot(q_diff)
        # Compute grad.
        grad_q = q_diff
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v