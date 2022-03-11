import time
from pathlib import Path

import numpy as np
import os

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.tet_mesh import tetrahedralize, read_tetgen_file, generate_tet_mesh, tet2obj
from py_diff_pd.common.tri_mesh import generate_tri_mesh
from py_diff_pd.common.tet_mesh import get_contact_vertex as get_tet_contact_vertex
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.display import export_gif
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

class DuckEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        youngs_modulus = 4e7
        poissons_ratio = 0.45
        state_force_parameters = options['state_force_parameters']
        center = ndarray(options['center'])
        start_degree = float(options['start_degree'])
        end_degree = float(options['end_degree'])
        start_rad = np.deg2rad(start_degree)
        end_rad = np.deg2rad(end_degree)
        radius = float(options['radius'])
        initial_degree = float(options['initial_degree'])
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
        c_start, s_start = np.cos(start_rad), np.sin(end_rad)
        # -s * x + c * z - c * height = 0.
        deformable.AddStateForce('arc_contact', [
            center[0], center[1], center[2],
            0, 1, 0,
            c_start, 0, -s_start,
            radius,
            end_rad - start_rad,
            3, kn, kf, mu])
        deformable.AddStateForce('planar_contact', [0, 0, 1, 0, 3, 1e3, 0.1, 1e4])

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Initial conditions.
        dofs = deformable.dofs()
        q0 = ndarray(mesh.py_vertices()).reshape((-1, 3))
        x_min = np.min(q0[:, 0])
        x_max = np.max(q0[:, 0])
        x_offset = np.max([-x_min, x_max])
        radius_adjusted = np.sqrt(radius ** 2 - x_offset ** 2)
        # Place q0 to the right location.
        init_rad = np.deg2rad(initial_degree)
        c, s = np.cos(init_rad), np.sin(init_rad)
        q0 = q0 @ ndarray([
            [s, 0, -c],
            [0, 1, 0],
            [c, 0, s]
        ]).T
        q0 += center + radius_adjusted * ndarray([c, 0, -s])
        q0 = q0.ravel()
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Create a trimesh obstacle.
        tri_mesh_name = Path(folder) / 'obs.obj'
        obs_seg_num = 50
        verts = []
        eles = []
        obs_seg_angle = (end_rad - start_rad) / obs_seg_num
        for i in range(obs_seg_num + 1):
            c, s = np.cos(start_rad + obs_seg_angle * i), np.sin(start_rad + obs_seg_angle * i)
            obs_pts_front = ndarray([
                c * radius + center[0],
                center[1] - 1.25,
                -s * radius + center[2],
            ])
            obs_pts_back = obs_pts_front + ndarray([0, 2.5, 0])
            obs_pts_front_down = obs_pts_front.copy()
            obs_pts_front_down[2] = -1
            obs_pts_back_down = obs_pts_back.copy()
            obs_pts_back_down[2] = -1
            verts += [obs_pts_front, obs_pts_back, obs_pts_front_down, obs_pts_back_down]
        for i in range(obs_seg_num):
            eles += [(4 * i, 4 * i + 1, 4 * i + 4),
                (4 * i + 4, 4 * i + 1, 4 * i + 5),
                (4 * i + 2, 4 * i, 4 * i + 4),
                (4 * i + 2, 4 * i + 4, 4 * i + 6),
                (4 * i + 1, 4 * i + 3, 4 * i + 5),
                (4 * i + 5, 4 * i + 3, 4 * i + 7)]
        eles += [(0, 2, 1),
            (2, 3, 1),
            (4 * obs_seg_num, 4 * obs_seg_num + 1, 4 * obs_seg_num + 2),
            (4 * obs_seg_num + 2, 4 * obs_seg_num + 1, 4 * obs_seg_num + 3),
            (3, 2, 4 * obs_seg_num + 3),
            (2, 4 * obs_seg_num + 2, 4 * obs_seg_num + 3)]
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
        self.__target = ndarray(options['target'])

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
            'camera_pos': (-1.0, -1.0, 0.9),
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

        renderer.render(light_rgb=(.5, .5, .5))

    def _loss_and_grad(self, q, v):
        q = ndarray(q).copy().reshape((-1, 3))
        loss = np.mean(q[:, 0])
        q_center = np.mean(q, axis=0)
        q_diff = q_center - self.__target
        loss = 0.5 * q_diff.dot(q_diff)
        grad_q = np.zeros(q.size).reshape((-1, 3))
        grad_q = np.ones((q.shape[0], 1)) @ q_diff.reshape((1, 3)) / q.shape[0]
        grad_q = grad_q.ravel()
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v
