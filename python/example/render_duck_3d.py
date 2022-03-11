import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle

from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.tri_mesh import generate_tri_mesh
from py_diff_pd.env.duck_env_3d import DuckEnv3d
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.display import export_mp4

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    folder = Path('render_duck_3d')
    center = ndarray([-0.6, 0, 7.7])
    start_deg = 60
    end_deg = 100
    init_deg = 70
    radius = 7.5
    init_speed = 10
    target = ndarray([-7.85, 0, 0.315])
    env = DuckEnv3d(seed, folder, {
        'state_force_parameters': [0, 0, -9.81, 1e5, 0.025, 1e4],
        'center': center,
        'start_degree': start_deg,
        'end_degree': end_deg,
        'initial_degree': init_deg,
        'radius': radius,
        'target': target })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 4000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 4000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    # methods = ('pd_eigen', 'newton_pcg', 'newton_cholesky')
    # opts = (pd_opt, newton_opt, newton_opt)
    methods = ('pd_eigen',)
    opts = (pd_opt,)

    dt = 5e-3
    frame_num = 200

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs).reshape((-1, 3))
    v0 += ndarray([-np.sin(np.deg2rad(init_deg)), 0, -np.cos(np.deg2rad(init_deg))]) * init_speed
    v0 = v0.ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    def variable_to_env(x):
        x = ndarray(x).copy().ravel()
        assert len(x) == 3
        kn = 10 ** x[0]
        kf = x[1]
        mu = 10 ** x[2]
        env = DuckEnv3d(seed, folder, {
            'state_force_parameters': [0, 0, -9.81, kn, kf, mu],
            'center': center,
            'start_degree': start_deg,
            'end_degree': end_deg,
            'initial_degree': init_deg,
            'radius': radius,
            'target': target })
        return env

    # Load or generate data.
    data = pickle.load(open('duck_3d/data_{:04d}_threads.bin'.format(thread_ct), 'rb'))
    for method, opt in zip(methods, opts):
        create_folder(folder / '{}/init'.format(method), exist_ok=True)
        if not (folder / method / 'init.data').is_file():
            x_init = data[method][0]['x']
            init_env = variable_to_env(x_init)
            _, info = init_env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder='{}/init'.format(method))
            pickle.dump(info, open(folder / method / 'init.data', 'wb'))

        create_folder(folder / '{}/final'.format(method), exist_ok=True)
        if not (folder / method / 'final.data').is_file():
            x_final = data[method][-1]['x']
            final_env = variable_to_env(x_final)
            _, info = final_env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=None)
            pickle.dump(info, open(folder / method / 'final.data', 'wb'))

    # Create paper figures.
    for method in methods:
        for name in ['init', 'final']:
            img_file = folder / method / '{}_overlapped.png'.format(name)
            options = {
                'file_name': img_file,
                'light_map': 'uffizi-large.exr',
                'sample': 4,
                'max_depth': 2,
                'camera_pos': (-1.3, -1.2, 1.0),
                'camera_lookat': (-0.2, 0.2, -0.1),
                'resolution': (1600, 1000),
                'fov': 27.7,
            }
            renderer = PbrtRenderer(options)
            renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
                transforms=[('s', 200)], color=(.4, .4, .4))
            renderer.add_tri_mesh(folder / 'obs.obj', transforms=[('s', 0.1)], color=(.3, .3, .3))

            # Plot the goal location.
            goal_inner_radius = 1.0
            goal_outer_radius = 1.2
            circle_num = 64
            dc = np.pi * 2 / circle_num
            verts = []
            eles = []
            for i in range(circle_num):
                c, s = np.cos(dc * i), np.sin(dc * i)
                vi = ndarray([c, s, 0]) * goal_inner_radius
                vo = ndarray([c, s, 0]) * goal_outer_radius
                verts.append(vi)
                verts.append(vo)
                ei0 = 2 * i
                ei1 = ei0 + 2 if i < circle_num - 1 else 0
                eo0 = ei0 + 1
                eo1 = ei1 + 1
                eles.append((ei0, eo0, eo1))
                eles.append((ei0, eo1, ei1))

            verts = ndarray(verts) + target * ndarray([1, 1, 0])
            eles = np.asarray(eles, dtype=int)
            generate_tri_mesh(verts, eles, folder / 'target.obj')
            renderer.add_tri_mesh(folder / 'target.obj', transforms=[('t', (0, 0, 0.01)), ('s', 0.1)], color=(.9, .9, .9))

            for i in range(0, frame_num + 1, 50):
                mesh_file = folder / method / name / '{:04d}.bin'.format(i)
                mesh = TetMesh3d()
                mesh.Initialize(str(mesh_file))
            
                renderer.add_tri_mesh(mesh, transforms=[('s', 0.1)], render_tet_edge=True, color=[1., .8, .0])

            renderer.render(light_rgb=(.5, .5, .5), verbose=True)

    # Render.
    for method in methods:
        for name in ['init', 'final']:
            for i in range(frame_num + 1):
                img_file = folder / method / name / '{:04d}.png'.format(i)
                mesh_file = folder / method / name / '{:04d}.bin'.format(i)
                mesh = TetMesh3d()
                mesh.Initialize(str(mesh_file))

                options = {
                    'file_name': img_file,
                    'light_map': 'uffizi-large.exr',
                    'sample': 4,
                    'max_depth': 2,
                    'camera_pos': (-1.3, -1.2, 1.0),
                    'camera_lookat': (-0.2, 0.2, -0.1),
                    'resolution': (1600, 1000),
                    'fov': 27.7,
                }
                renderer = PbrtRenderer(options)

                renderer.add_tri_mesh(mesh, transforms=[('s', 0.1)], render_tet_edge=True, color=[1., .8, .0])
                renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
                    transforms=[('s', 200)], color=(.4, .4, .4))
                renderer.add_tri_mesh(folder / 'obs.obj', transforms=[('s', 0.1)], color=(.3, .3, .3))

                # Plot the goal location.
                goal_inner_radius = 1.0
                goal_outer_radius = 1.2
                circle_num = 64
                dc = np.pi * 2 / circle_num
                verts = []
                eles = []
                for i in range(circle_num):
                    c, s = np.cos(dc * i), np.sin(dc * i)
                    vi = ndarray([c, s, 0]) * goal_inner_radius
                    vo = ndarray([c, s, 0]) * goal_outer_radius
                    verts.append(vi)
                    verts.append(vo)
                    ei0 = 2 * i
                    ei1 = ei0 + 2 if i < circle_num - 1 else 0
                    eo0 = ei0 + 1
                    eo1 = ei1 + 1
                    eles.append((ei0, eo0, eo1))
                    eles.append((ei0, eo1, ei1))

                verts = ndarray(verts) + target * ndarray([1, 1, 0])
                eles = np.asarray(eles, dtype=int)
                generate_tri_mesh(verts, eles, folder / 'target.obj')
                renderer.add_tri_mesh(folder / 'target.obj', transforms=[('t', (0, 0, 0.01)), ('s', 0.1)], color=(.9, .9, .9))

                renderer.render(light_rgb=(.5, .5, .5), verbose=True)

    for method in methods:
        for name in ['init', 'final']:
            export_mp4(folder / method / name, folder / method / '{}.mp4'.format(name), fps=100)