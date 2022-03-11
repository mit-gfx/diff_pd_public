import time
from pathlib import Path

import numpy as np
import os

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.tet_mesh import tetrahedralize
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import print_info, create_folder, ndarray
from py_diff_pd.common.display import export_mp4
from py_diff_pd.common.tet_mesh import generate_tet_mesh, read_tetgen_file
from py_diff_pd.common.tet_mesh import get_contact_vertex as get_tet_contact_vertex
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

class ArmadilloEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        # Generate armadillo mesh.
        ele_file_name = Path(root_path) / 'asset' / 'mesh' / 'armadillo_10k.ele'
        node_file_name = Path(root_path) / 'asset' / 'mesh' / 'armadillo_10k.node'
        verts, eles = read_tetgen_file(node_file_name, ele_file_name)
        # To make the mesh consistent with our coordinate system, we need to:
        # - rotate the model along +x by 90 degrees.
        # - shift it so that its min_z = 0.
        # - divide it by 1000.
        R = ndarray([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        verts = verts @ R.T
        # Next, rotate along z by 180 degrees.
        R = ndarray([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ])
        verts = verts @ R.T
        min_z = np.min(verts, axis=0)[2]
        verts[:, 2] -= min_z
        verts /= 1000
        tmp_bin_file_name = '.tmp.bin'
        generate_tet_mesh(verts, eles, tmp_bin_file_name)
        mesh = TetMesh3d()
        mesh.Initialize(str(tmp_bin_file_name))
        deformable = TetDeformable()
        deformable.Initialize(tmp_bin_file_name, density, 'neohookean', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)

        # Boundary conditions.
        # Figure out the lowest z nodes.
        vert_num = mesh.NumOfVertices()
        all_verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        max_corner = np.max(all_verts, axis=0)
        min_corner = np.min(all_verts, axis=0)
        center = (max_corner + min_corner) / 2
        min_z = min_corner[2]
        max_z = max_corner[2]
        min_x = min_corner[0]
        max_x = max_corner[0]
        dirichlet_dofs = []
        self.__min_x_nodes = []
        self.__max_x_nodes = []
        for i in range(vert_num):
            vx, vy, vz = all_verts[i]
            if vx - min_x < 1e-3:
                self.__min_x_nodes.append(i)
            if max_x - vx < 1e-3:
                self.__max_x_nodes.append(i)
            if vz - min_z < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)
                dirichlet_dofs += [3 * i, 3 * i + 1, 3 * i + 2]
            if max_z - vz < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)
                dirichlet_dofs += [3 * i + 2,]
        self.__dirichlet_dofs = dirichlet_dofs
        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)

        # Initial state by rotating the armadillo.
        q0 = np.copy(all_verts)
        theta = float(options['init_rotate_angle'])
        for i in range(vert_num):
            vi = all_verts[i]
            th = (vi[2] - min_z) / (max_corner[2] - min_z) * theta
            c, s = np.cos(th), np.sin(th)
            R = ndarray([[c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])
            q0[i] = R @ (vi - center) + center

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = q0.ravel()
        v0 = ndarray(np.zeros(dofs)).ravel()
        f_ext = ndarray(np.zeros(dofs)).ravel()

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

        self.__spp = options['spp'] if 'spp' in options else 4

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        # This (0, 2) shape is due to the usage of Neohookean materials.
        return np.zeros((0, 2))

    def is_dirichlet_dof(self, dof):
        return dof in self.__dirichlet_dofs

    def _display_mesh(self, mesh_file, file_name):
        # Size of the bounding box: [-0.06, -0.05, 0] - [0.06, 0.05, 0.14]
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.12, -0.8, 0.34),
            'camera_lookat': (0, 0, .15)
        }
        renderer = PbrtRenderer(options)

        mesh = TetMesh3d()
        mesh.Initialize(mesh_file)
        vert_num = mesh.NumOfVertices()
        all_verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        renderer.add_tri_mesh(mesh, color='0096c7',
            transforms=[('s', 2)],
            render_tet_edge=True,
        )
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render()

    def min_x_nodes(self):
        return self.__min_x_nodes

    def max_x_nodes(self):
        return self.__max_x_nodes

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)
