import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.tri_mesh import generate_circle_mesh, get_boundary_edge
from py_diff_pd.common.display import display_tri_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import TriMesh2d, TriDeformable, StdRealVector

class CircleEnv2d(EnvBase):
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
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, -9.81])

        # Mesh parameters.
        origin = ndarray([0, 0.2])
        radius = 0.1
        radius_bin_num = 2 * refinement
        angle_bin_num = 6 * refinement
        bin_file_name = str(folder / 'mesh.bin')
        generate_circle_mesh(origin, radius, radius_bin_num, angle_bin_num, bin_file_name)
        mesh = TriMesh2d()
        mesh.Initialize(bin_file_name)

        # FEM parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e4
        deformable = TriDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)

        # External force.
        deformable.AddStateForce('gravity', state_force_parameters)

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Collision.
        boundary_edges = get_boundary_edge(mesh)
        contact_nodes = set()
        for vi, vj in boundary_edges:
            contact_nodes.add(vi)
            contact_nodes.add(vj)
        contact_nodes = list(contact_nodes)
        deformable.AddPdEnergy('planar_collision', [1e4, 0.0, 1.0, 0], contact_nodes)

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
        mesh = TriMesh2d()
        mesh.Initialize(mesh_file)
        display_tri_mesh(mesh, xlim=[-0.3, 0.3], ylim=[-0.2, 0.4],
            file_name=file_name, show=False)

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)