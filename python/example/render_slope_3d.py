import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle

from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.env.slope_env_3d import SlopeEnv3d
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    folder = Path('render_slope_3d')
    env = SlopeEnv3d(seed, folder, {
        'state_force_parameters': [0, 0, -9.81, 1e5, 0.025, 1e4],
        'slope_degree': 20,
        'initial_height': 2.0 })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 4000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 4000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('pd_eigen', 'newton_pcg', 'newton_cholesky')
    opts = (pd_opt, newton_opt, newton_opt)

    dt = 5e-3
    frame_num = 400

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    def variable_to_env(x):
        x = ndarray(x).copy().ravel()
        assert len(x) == 3
        kn = 10 ** x[0]
        kf = x[1]
        mu = 10 ** x[2]
        env = SlopeEnv3d(seed, folder, {
            'state_force_parameters': [0, 0, -9.81, kn, kf, mu],
            'slope_degree': 20,
            'initial_height': 2.0,
            'spp': 1 })
        return env

    for kf in [0.1, 0.3, 0.5, 1.0]:
        env = variable_to_env([4, kf, 3])
        info, _ = env.simulate(dt, frame_num, 'pd_eigen', pd_opt, q0, v0, a0, f0, require_grad=False,
            vis_folder='{}'.format(kf), render_frame_skip=200)
        pickle.dump(info, open(folder / '{}.data'.format(kf), 'wb'))

    # Now the rendering script.
    options = {
        'file_name': folder / 'overlap.png',
        'light_map': 'uffizi-large.exr',
        'sample': 256,
        'max_depth': 2,
        'camera_pos': (-2.4, -0.26, 0.64),
        'camera_lookat': (0, -0.26, 0),
        'resolution': (1024, 768)
    }
    renderer = PbrtRenderer(options)
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
        transforms=[('r', (-np.pi / 2, 0, 0, 1)), ('t', (0.2, 0, -0.05)), ('s', 200)], color=(.4, .4, .4))

    mesh_file = folder / '0.1/0000.bin'
    mesh = TetMesh3d()
    mesh.Initialize(str(mesh_file))
    renderer.add_tri_mesh(mesh, transforms=[('t', (0, 2.6, 0)), ('s', 0.1)], render_tet_edge=True, color=[1., .8, .0])
    renderer.add_tri_mesh(folder / 'obs.obj', transforms=[('t', (0, 2.6, 0)), ('s', 0.1)], color=(.3, .3, .3))

    for i, kf in enumerate(reversed([0.1, 0.3, 0.5, 1.0])):
        # Load ducks.
        mesh_file = folder / '{}'.format(kf) / '0400.bin'
        mesh = TetMesh3d()
        mesh.Initialize(str(mesh_file))
        renderer.add_tri_mesh(mesh, transforms=[('t', (0, -i * 2.6, 0)), ('s', 0.1)], render_tet_edge=True, color=[1., .8, .0])
        renderer.add_tri_mesh(folder / 'obs.obj', transforms=[('t', (0, -i * 2.6, 0)), ('s', 0.1)], color=(.3, .3, .3))

    renderer.render(light_rgb=(.5, .5, .5), verbose=True)