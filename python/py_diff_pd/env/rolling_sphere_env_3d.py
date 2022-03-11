import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_contact_vertex
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector

class RollingSphereEnv3d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options['refinement']
        youngs_modulus = options['youngs_modulus']
        poissons_ratio = options['poissons_ratio']
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3

        # Shape of the rolling sphere.
        cell_nums = (refinement, refinement, refinement)
        origin = ndarray([0, 0, 0])
        node_nums = [n + 1 for n in cell_nums]
        voxels = np.ones(cell_nums)
        radius = 0.05
        dx = radius * 2 / refinement
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        for i in range(cell_nums[0]):
            for j in range(cell_nums[1]):
                for k in range(cell_nums[2]):
                    cell_center = ndarray([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                    if np.linalg.norm(cell_center - ndarray([radius, radius, radius])) > radius:
                        voxels[i][j][k] = 0
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
        friction_node_idx = get_contact_vertex(mesh)
        # Uncomment the code below if you would like to display the contact set for a sanity check:
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in friction_node_idx])
        ax.scatter(v[:, 0], v[:, 1], v[:, 2])
        plt.show()
        '''

        # Friction_node_idx = all vertices on the edge.
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)

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
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.__radius = radius
        self.__contact_dofs = len(friction_node_idx) * 3

        self.__spp = options['spp'] if 'spp' in options else 4

    def radius(self):
        return self.__radius

    def contact_dofs(self):
        return self.__contact_dofs

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def _display_mesh(self, mesh_file, file_name):
        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        render_hex_mesh(mesh, file_name=file_name,
            resolution=(400, 400), sample=self.__spp,
            camera_pos=(2, -2.2, 1.5),
            transforms=[
                ('t', (-0.05, 0, 0)),
                ('s', 6)
            ], render_voxel_edge=True)

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)