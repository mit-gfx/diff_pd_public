import time
from pathlib import Path

import numpy as np
import os

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.tet_mesh import tetrahedralize, read_tetgen_file, generate_tet_mesh, tet2obj
from py_diff_pd.common.tri_mesh import generate_tri_mesh
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.display import export_gif
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

class SlopeEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        youngs_modulus = 4e6
        poissons_ratio = 0.45
        state_force_parameters = options['state_force_parameters']
        slope_degree = float(options['slope_degree'])
        initial_height = float(options['initial_height'])
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3

        # Mesh parameters.
        tmp_bin_file_name = '.tmp.bin'
        obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'bob.obj'
        verts, eles = tetrahedralize(obj_file_name, normalize_input=False, options={ 'minratio': 1.1 })
        generate_tet_mesh(verts, eles, tmp_bin_file_name)
        mesh = TetMesh3d()
        mesh.Initialize(str(tmp_bin_file_name))
        deformable = TetDeformable()
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)

        # External force.
        g = state_force_parameters[:3]
        deformable.AddStateForce('gravity', g)

        # Contact.
        kn, kf, mu = state_force_parameters[3:]
        # The line equation:
        c, s = np.cos(np.deg2rad(slope_degree)), np.sin(np.deg2rad(slope_degree))
        # -s * x + c * z - c * height = 0.
        deformable.AddStateForce('planar_contact', [-s, 0, c, -c * initial_height, 3, kn, kf, mu])

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Initial conditions.
        dofs = deformable.dofs()
        q0 = ndarray(mesh.py_vertices()).reshape((-1, 3))
        # Place q0 to the right location.
        q0 = q0 @ ndarray([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ]).T
        q0 += ndarray([0, 0, initial_height])
        q0 = q0.ravel()
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Create a trimesh obstacle.
        tri_mesh_name = Path(folder) / 'obs.obj'
        # Plane equation: -s * x + c * z - c * height = 0.
        verts = ndarray([
            [(-c - c * initial_height) / s, -1.25, -1],
            [(-c - c * initial_height) / s, 1.25, -1],
            [1, -1.25, (1 * s + c * initial_height) / c],
            [1, 1.25, (1 * s + c * initial_height) / c],
            [1, -1.25, -1],
            [1, 1.25, -1]
        ])
        eles = [[1, 0, 2],
            [2, 3, 1],
            [5, 1, 3],
            [2, 0, 4],
            [4, 0, 1],
            [1, 5, 4],
            [3, 2, 5],
            [5, 2, 4]]
        generate_tri_mesh(verts, eles, tri_mesh_name)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__spp = int(options['spp']) if 'spp' in options else 4

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (-0.52, -.52, 0.44),
            'camera_lookat': (-0.05, 0, 0.05),
            'resolution': (1024, 768)
        }
        renderer = PbrtRenderer(options)

        mesh = TetMesh3d()
        mesh.Initialize(mesh_file)
        vert_num = mesh.NumOfVertices()
        renderer.add_tri_mesh(mesh, transforms=[('s', 0.1)], render_tet_edge=True, color=[1., .8, .0])
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            transforms=[('s', 200)], color=(.4, .4, .4))
        renderer.add_tri_mesh(self._folder / 'obs.obj', transforms=[('s', 0.1)], color=(.3, .3, .3))

        renderer.render(light_rgb=(.5, .5, .5), verbose=True)

    def _loss_and_grad(self, q, v):
        q = ndarray(q).copy().reshape((-1, 3))
        loss = np.mean(q[:, 0])
        grad_q = np.zeros(q.size).reshape((-1, 3))
        grad_q[:, 0] = 1 / len(q[:, 0])
        grad_q = grad_q.ravel()
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v
