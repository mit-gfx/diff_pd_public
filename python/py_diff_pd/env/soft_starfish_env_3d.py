import time
from pathlib import Path
import os

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.tet_mesh import generate_tet_mesh, tet2obj, tetrahedralize, get_boundary_face
from py_diff_pd.common.tet_mesh import get_contact_vertex as get_tet_contact_vertex
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

class SoftStarfishEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        # See this file for the material parameters:
        # https://www.smooth-on.com/tb/files/SOMA_FOAMA_TB.pdf
        # We are using soma foma 25.
        # Density seems to be 400 kg/m3.
        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 5e5
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.4
        act_stiffness = options['act_stiffness'] if 'act_stiffness' in options else 2e6
        enable_y_act = bool(options['y_actuator']) if 'y_actuator' in options else True
        enable_z_act = bool(options['z_actuator']) if 'z_actuator' in options else True
        fix_center_x = bool(options['fix_center_x']) if 'fix_center_x' in options else True
        fix_center_y = bool(options['fix_center_y']) if 'fix_center_y' in options else True
        fix_center_z = bool(options['fix_center_z']) if 'fix_center_z' in options else True
        self._stepwise_loss = bool(options['use_stepwise_loss']) if 'use_stepwise_loss' in options else True
        self.__data = options['data']
        self.__substep = int(options['substep'])
        self.__render_markers = bool(options['render_markers']) if 'render_markers' in options else True

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 4e2
        tmp_bin_file_name = '.tmp.bin'
        obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'starfish_half_simplified.obj'
        verts, eles = tetrahedralize(obj_file_name, normalize_input=False, options={ 'minratio': 1.1 })
        generate_tet_mesh(verts, eles, tmp_bin_file_name)
        mesh = TetMesh3d()
        mesh.Initialize(str(tmp_bin_file_name))
        deformable = TetDeformable()
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Hydrodynamic forces.
        v_rho = 1e3
        v_water = ndarray([0, 0, 0])    # Velocity of the water.
        # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
        Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
        # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
        Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
        # The current Cd and Ct are similar to Figure 2 in SoftCon.
        # surface_faces is a list of (v0, v1, v2) where v0, v1, v2 are the vertex indices of the corners of a boundary face.
        # The order of v0, v1, v2 follows the right-hand rule: if your right hand follows v0 -> v1 -> v2, your thumb will
        # point to the outward normal.
        surface_faces = get_boundary_face(mesh)
        deformable.AddStateForce('hydrodynamics', np.concatenate([[v_rho,], v_water, Cd_points.ravel(), Ct_points.ravel(),
            ndarray(surface_faces).ravel()]))

        # Dirichlet boundary conditions.
        # The starfish is swimming along the -x axis.
        # Pick the center of the starfish and fix their y and z locations.
        half_limb_width = 7.5 / 1000
        limb_length = (70 - 18) / 1000
        half_center_size = 18 / 1000
        vert_num = mesh.NumOfVertices()
        central_vertices = []
        for i in range(vert_num):
            _, y, z = ndarray(mesh.py_vertex(i))
            if np.abs(y) < half_center_size and np.abs(z) < half_center_size:
                central_vertices.append(i)
        dirichlet_dofs = []
        for i in central_vertices:
            x, y, z = ndarray(mesh.py_vertex(i))
            if fix_center_x:
                dirichlet_dofs.append(3 * i)
                deformable.SetDirichletBoundaryCondition(3 * i, x)
            if fix_center_y:
                dirichlet_dofs.append(3 * i + 1)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, y)
            if fix_center_z:
                dirichlet_dofs.append(3 * i + 2)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, z)

        # Actuators.
        # Range of the center of the starfish: -18mm to +18mm.
        # Width of the limb: -7.5mm to +7.5mm.
        half_limb_width = 7.5 / 1000
        half_center_size = 18 / 1000
        ele_num = mesh.NumOfElements()
        y_pos_act_eles = []
        y_neg_act_eles = []
        z_pos_act_eles = []
        z_neg_act_eles = []
        for ei in range(ele_num):
            v_idx = list(mesh.py_element(ei))
            v = [ndarray(mesh.py_vertex(i)) for i in v_idx]
            v = ndarray(v)
            # A trick for starfish_half_simplified.obj:
            # Actuators must be defined on tet elements on the flat plane.
            if np.min(np.abs(v[:, 0])) > 1e-3: continue
            # Compute the average location.
            v_com = np.mean(v, axis=0)
            x, y, z = v_com
            if np.abs(y) > half_limb_width and np.abs(z) > half_limb_width: continue
            if y < -half_center_size:
                y_neg_act_eles.append(ei)
            elif y > half_center_size:
                y_pos_act_eles.append(ei)
            elif z < -half_center_size:
                z_neg_act_eles.append(ei)
            elif z > half_center_size:
                z_pos_act_eles.append(ei)

        if enable_y_act:
            deformable.AddActuation(act_stiffness, ndarray([0, 1, 0]), y_neg_act_eles)
            deformable.AddActuation(act_stiffness, ndarray([0, 1, 0]), y_pos_act_eles)
        if enable_z_act:
            deformable.AddActuation(act_stiffness, ndarray([0, 0, 1]), z_neg_act_eles)
            deformable.AddActuation(act_stiffness, ndarray([0, 0, 1]), z_pos_act_eles)

        # Marker positions.
        # The four markers are on the x = 0 plane, from -z to +z, and y < 0.
        markers_info = {}
        def find_marker_idx(z_coord):
            dist = 0.004
            min_y = np.inf
            min_idx = None
            for i in range(vert_num):
                x, y, z = ndarray(mesh.py_vertex(i))
                if np.abs(x) > 1e-3: continue
                if z_coord - dist < z < z_coord + dist:
                    if y < min_y:
                        min_y = y
                        min_idx = i
            return min_idx
        m1_idx = find_marker_idx(-half_center_size - limb_length)
        m2_idx = find_marker_idx(-half_center_size - limb_length + 30 / 1000)
        m3_idx = find_marker_idx(-half_center_size - limb_length + (30 + 76.2) / 1000)
        m4_idx = find_marker_idx(-half_center_size - limb_length + (30 + 76.2 + 27) / 1000)
        markers_info['M1'] = m1_idx
        markers_info['M2'] = m2_idx
        markers_info['M3'] = m3_idx
        markers_info['M4'] = m4_idx

        # Uncomment the code below to plot the central vertices.
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in central_vertices])
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], marker='+')
        v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in range(vert_num)])
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], marker='o', alpha=0.1)
        # Plot the four actuators.
        def plot_actuator(e_indices, color, label):
            v_indices = set()
            for ei in e_indices:
                v_idx = list(mesh.py_element(ei))
                for vi in v_idx:
                    v_indices.add(vi)
            v = []
            for vi in v_indices:
                v.append(ndarray(mesh.py_vertex(vi)))
            v = ndarray(v)
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], color=color, label=label)
        plot_actuator(y_neg_act_eles, 'tab:red', 'y-neg actuator')
        plot_actuator(y_pos_act_eles, 'tab:green', 'y-pos actuator')
        plot_actuator(z_neg_act_eles, 'tab:cyan', 'z-neg actuator')
        plot_actuator(z_pos_act_eles, 'tab:pink', 'z-pos actuator')
        # Plot markers.
        for key in markers_info:
            idx = markers_info[key]
            v = ndarray(mesh.py_vertex(idx))
            ax.scatter(v[0], v[1], v[2], marker='*', s=100, label=key)
        max_offset = limb_length + half_center_size
        ax.scatter([-max_offset - 0.01, max_offset + 0.01],
            [-max_offset - 0.01, max_offset + 0.01],
            [-max_offset - 0.01, max_offset + 0.01])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([-max_offset, max_offset])
        ax.set_ylim([-max_offset, max_offset])
        ax.set_zlim([-max_offset, max_offset])
        ax.legend()
        plt.show()
        '''

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
        self.__dirichlet_dofs = dirichlet_dofs

        self.__spp = options['spp'] if 'spp' in options else 4
        self.__markers_info = markers_info
        # The full tendon length comes from Josie.
        self.__full_tendon_length = 65 / 1000

    def full_tendon_length(self):
        return self.__full_tendon_length

    def markers_info(self):
        return self.__markers_info

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return dof in self._dirichlet_dofs

    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.15, -0.75, .4),
            'camera_lookat': (0, .15, .4)
        }
        renderer = PbrtRenderer(options)

        mesh = TetMesh3d()
        mesh.Initialize(mesh_file)
        renderer.add_tri_mesh(mesh, color=(.6, .3, .2),
            transforms=[
                ('s', 1.5),
                ('t', (0.2, 0.4, 0.4))
            ],
            render_tet_edge=False,
        )

        # Render markers.
        if self.__render_markers:
            frame_idx = int(mesh_file.split('.')[0].split('/')[-1])
            for i in range(1, 5):
                name = 'M{:d}'.format(i)
                vi = self.__markers_info[name]
                pos = ndarray(mesh.py_vertex(vi))
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': pos,
                    'radius': 0.005
                },
                transforms=[
                    ('s', 1.5),
                    ('t', (0.2, 0.4, 0.4))
                ],
                color=(.2, .3, .7))
                _, yi, _ = pos
                xi_target = self.__data[name + '_rel_x'][int(frame_idx // self.__substep)] + self._q0[3 * vi]
                zi_target = self.__data[name + '_rel_z'][int(frame_idx // self.__substep)] + self._q0[3 * vi + 2]
                target_pos = ndarray([xi_target, yi, zi_target])
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': target_pos,
                    'radius': 0.005
                },
                transforms=[
                    ('s', 1.5),
                    ('t', (0.2, 0.4, 0.4))
                ],
                color=(.2, .7, .3))

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render(verbose=True)

    def _stepwise_loss_and_grad(self, q, v, i):
        loss = 0
        grad_q = np.zeros(q.size)
        grad_v = np.zeros(v.size)
        weights = { 'M1': 10.0, 'M2': 1.0, 'M3': 1.0, 'M4': 10.0 }
        for k in range(1, 5):
            name = 'M{:d}'.format(k)
            vi = self.__markers_info[name]
            xi = q[3 * vi]
            zi = q[3 * vi + 2]
            xi_target = self.__data[name + '_rel_x'][int(i // self.__substep)] + self._q0[3 * vi]
            zi_target = self.__data[name + '_rel_z'][int(i // self.__substep)] + self._q0[3 * vi + 2]
            w = weights[name]
            loss += w * (xi - xi_target) ** 2 + w * (zi - zi_target) ** 2
            grad_q[3 * vi] += 2 * w * (xi - xi_target)
            grad_q[3 * vi + 2] += 2 * w * (zi - zi_target)
        return loss, grad_q, grad_v

    def _loss_and_grad(self, q, v):
        # Compute the center of mass.
        com = np.mean(q.reshape((-1, 3)), axis=0)
        loss = com[0]
        # Compute grad.
        grad_q = np.zeros(q.size)
        vertex_num = int(q.size // 3)
        grad_q[::3] = 1.0 / vertex_num
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v
