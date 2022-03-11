import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.display import export_mp4
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.sim_to_real_env_3d import SimToRealEnv3d
from py_diff_pd.common.project_path import root_path

def overlap(photo_sequence, rendering_sequence, photo_alpha, rendering_alpha, output_sequence):
    photo_data = []
    rendering_data = []
    for photo_name in photo_sequence:
        with cbook.get_sample_data(photo_name) as f:
            img = plt.imread(f)
        photo_data.append(ndarray(img[:, :, :3]))
    for rendering_name in rendering_sequence:
        with cbook.get_sample_data(rendering_name) as f:
            img = plt.imread(f)
        rendering_data.append(ndarray(img[:, :, :3]))
    for p, r, f in zip(photo_data, rendering_data, output_sequence):
        output_img = photo_alpha * p + rendering_alpha * r
        plt.imsave(f, output_img)

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('sim_to_real_experiment')

    motion_data_folder = Path(root_path) / 'python/example/sim_to_real_calibration/experiment'
    substeps = 4
    # This is the frame we estimate the object starts rolling. Could be outside the range of the camera.
    start_frame = 160
    # This is typically the last frame in the `python/example/sim_to_real_calibration/experiment/` folder.
    end_frame = 210
    # The video was taken at 60 fps.
    dt = (1 / 60) / substeps
    frame_num = (end_frame - start_frame) * substeps
    # We estimate from the video that the initial force is applied for roughly 10 frames.
    init_force_frame_num = 10 * substeps
    # Optimization parameters.
    thread_ct = 6
    opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    method = 'pd_eigen'

    # Build an environment.
    def get_env(camera_parameters):
        camera_parameters = ndarray(camera_parameters).copy().ravel()
        camera_pos = camera_parameters[:3]
        camera_yaw, camera_pitch, camera_alpha = camera_parameters[3:6]
        env = SimToRealEnv3d(folder, {
            'camera_pos': camera_pos,
            'camera_yaw': camera_yaw,
            'camera_pitch': camera_pitch,
            'camera_alpha': camera_alpha,
            'experiment_folder': motion_data_folder,
            'img_height': 720,
            'img_width': 1280,
            'substeps': substeps,
            'start_frame': start_frame
        })
        return env

    # Obtain some educated guess of the camera parameters.
    camera_data_files = Path(motion_data_folder).glob('*.data')
    all_alpha = []
    for f in camera_data_files:
        info = pickle.load(open(f, 'rb'))
        all_alpha.append(info['alpha'])
        all_alpha.append(info['beta'])
    init_camera_alpha = np.mean(all_alpha)
    init_camera_pos = ndarray([0, 0, 0])
    init_camera_yaw = 0
    init_camera_pitch = 0
    init_env = get_env(np.concatenate([init_camera_pos, [init_camera_yaw, init_camera_pitch, init_camera_alpha]]))
    deformable = init_env.deformable()

    # Build the initial state of the object.
    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    v0 = np.zeros(dofs)
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]

    # Determine initial q --- this is based on the decision variables.
    # Our decision variables include the following:
    # - init_com: 3D. The center of mass.
    # - init_yaw: 1D.
    # - init_f: 3D. The initial force applied to force_node_idx below.
    # - camera_pos: 3D. The location of the camera.
    # - camera_yaw: 1D.
    # - camera_pitch: 1D.
    # - camera_alpha: 1D.
    q0 = init_env.default_init_position()
    q0_offset = q0.reshape((-1, 3)) - np.mean(q0.reshape((-1, 3)), axis=0)
    def get_init_q(x):
        x = ndarray(x).copy().ravel()
        init_com = x[:3]
        init_yaw = x[3]
        c_yaw, s_yaw = np.cos(init_yaw), np.sin(init_yaw)
        init_R = ndarray([
            [c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]
        ])
        init_q = (q0_offset @ init_R.T + init_com).ravel()
        return ndarray(init_q).copy()

    def get_init_q_gradient(x, grad_init_q):
        x = ndarray(x).copy().ravel()
        grad = np.zeros(x.size)
        # Init com.
        grad[:3] = np.sum(grad_init_q.reshape((-1, 3)), axis=0)
        # Init yaw.
        init_yaw = x[3]
        c_yaw, s_yaw = np.cos(init_yaw), np.sin(init_yaw)
        dR = ndarray([
            [-s_yaw, -c_yaw, 0],
            [c_yaw, -s_yaw, 0],
            [0, 0, 0]
        ])
        grad[3] = (q0_offset @ dR.T).ravel().dot(grad_init_q)
        return ndarray(grad).copy()

    # Determine external force f0 --- this is also based on the decision variables.
    # Find the top q locations to apply external forces.
    q_max_z = np.max(q0_offset[:, 2])
    force_node_idx = []
    for i in range(q0_offset.shape[0]):
        if np.abs(q0_offset[i, 2] - q_max_z) < 1e-5:
            force_node_idx.append(i)

    def get_external_f(x):
        x = ndarray(x).copy().ravel()
        init_f = x[4:7] / 1000  # Convert from Newton to mN.
        f0 = [np.zeros(dofs) for _ in range(frame_num)]
        for i in range(init_force_frame_num):
            fi = f0[i].reshape((-1, 3))
            fi[force_node_idx] = init_f
            f0[i] = ndarray(fi).copy().ravel()
        return f0

    def get_external_f_gradient(x, grad_f):
        x = ndarray(x).copy().ravel()
        grad = np.zeros(x.size)
        for i in range(init_force_frame_num):
            dfi = grad_f[i].reshape((-1, 3))
            grad[4:7] += np.sum(dfi[force_node_idx], axis=0) / 1000
        return ndarray(grad).copy()

    # Optimization.
    # Variables to be optimized:
    # - init_com: 3D. The center of mass.
    # - init_yaw: 1D.
    # - init_f: 3D. The initial force applied to force_node_idx below. Note that we use mN instead of Newton as the unit.
    # - camera_pos: 3D. The location of the camera.
    # - camera_yaw: 1D.
    # - camera_pitch: 1D.
    # - camera_alpha: 1D.
    x_ref = ndarray(np.concatenate([
        [0.1, 0.2, 0.0, np.pi / 2, -3, 0, 0],
        init_camera_pos,
        [init_camera_yaw, init_camera_pitch, init_camera_alpha]
    ]))
    x_lb = ndarray(np.concatenate([
        [0.05, 0.15, 0.0, np.pi / 2 - 0.2, -3.5, -0.1, 0],
        [init_camera_pos[0], init_camera_pos[1] - 0.05, init_camera_pos[2] - 0.05],
        [init_camera_yaw - 0.2, init_camera_pitch - 0.2, init_camera_alpha - 300]
    ]))
    x_ub = ndarray(np.concatenate([
        [0.15, 0.25, 0.0, np.pi / 2 + 0.2, -2.5, 0.1, 0],
        [init_camera_pos[0], init_camera_pos[1] + 0.05, init_camera_pos[2] + 0.05],
        [init_camera_yaw + 0.2, init_camera_pitch + 0.2, init_camera_alpha + 300]
    ]))
    x_fixed = np.array([True, True, True, True, True, True, True, False, False, False, False, False, False])

    def render_x(x_reduced, vis_folder, render_frame_skip):
        # Visualize initial guess.
        x_full = ndarray(x_ref).copy().ravel()
        x_full[~x_fixed] = x_reduced
        env = get_env(x_full[7:13])
        init_q = get_init_q(x_full)
        init_f = get_external_f(x_full)
        env.simulate(dt, frame_num, method, opt, init_q, v0, a0, init_f, require_grad=False, vis_folder=vis_folder,
            render_frame_skip=render_frame_skip)

    data = pickle.load(open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'rb'))
    data = data[method]

    # Visualize the initial and best results.
    x_best_idx = np.argmin([d['loss'] for d in data])
    print_info('Initial loss: {:3.6f}; best loss: {:3.6f}'.format(data[0]['loss'], data[x_best_idx]['loss']))
    render_x(data[0]['x'], 'init', substeps)
    render_x(data[x_best_idx]['x'], 'best', substeps)

    # Overlap it with the real photos.
    photo_folder = Path(root_path) / 'python/example/sim_to_real_calibration/experiment_video'
    photo_sequence = [photo_folder / '{:04d}.png'.format(i) for i in range(start_frame, end_frame + 1)]
    init_sequence = [Path(root_path) / 'python/example' / folder / 'init' / '{:04d}.png'.format(i)
        for i in range(0, (end_frame - start_frame) * substeps, substeps)]
    best_sequence = [Path(root_path) / 'python/example' / folder / 'best' / '{:04d}.png'.format(i)
        for i in range(0, (end_frame - start_frame + 1) * substeps, substeps)]
    create_folder(folder / 'init_overlap', exist_ok=True)
    create_folder(folder / 'best_overlap', exist_ok=True)
    init_output_sequence = [Path(root_path) / 'python/example' / folder / 'init_overlap' / '{:04d}.png'.format(i)
        for i in range(start_frame, end_frame + 1)]
    best_output_sequence = [Path(root_path) / 'python/example' / folder / 'best_overlap' / '{:04d}.png'.format(i)
        for i in range(start_frame, end_frame + 1)]
    overlap(photo_sequence, init_sequence, 0.5, 0.5, init_output_sequence)
    overlap(photo_sequence, best_sequence, 0.5, 0.5, best_output_sequence)
    export_mp4(folder / 'init_overlap', folder / 'init_overlap.mp4', fps=10)
    export_mp4(folder / 'best_overlap', folder / 'best_overlap.mp4', fps=10)