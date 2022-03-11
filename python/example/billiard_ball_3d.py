import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle
import matplotlib.pyplot as plt

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.billiard_ball_env_3d import BilliardBallEnv3d
from py_diff_pd.common.project_path import root_path

img_height, img_width = 720, 1280
def pxl_to_cal(pxl):
    pxl = ndarray(pxl).copy()
    pxl[:, 1] *= -1
    pxl[:, 1] += img_height
    return pxl
def cal_to_pxl(cal):
    cal = ndarray(cal).copy()
    cal[:, 1] -= img_height
    cal[:, 1] *= -1
    return cal

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    folder = Path('billiard_ball_3d')

    # Simulation parameters.
    substeps = 3
    dt = (1 / 60) / substeps
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 2000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4,
        'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 2000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4,
        'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 }
    pd_method = 'pd_eigen'
    methods = ('newton_pcg', 'newton_cholesky', pd_method)
    opts = (newton_opt, newton_opt, pd_opt)

    # Extract the initial information of the balls.
    ball_radius = 0.06858 / 2   # In meters and from measurement/googling the diameter of a tennis ball.
    experiment_data_folder = Path(root_path) / 'python/example/billiard_ball_calibration/experiment'
    ball_xy_positions = pickle.load(open(experiment_data_folder / 'ball_xy_positions.data', 'rb'))
    start_frame = 150
    end_frame = 200
    frame_num = (end_frame - start_frame - 1) * substeps
    # Unlike in calibration, the height is set to be 0 here.
    ball_0_positions = [(pos[0, 0], pos[0, 1], 0) for _, pos in ball_xy_positions]
    ball_1_positions = [(pos[1, 0], pos[1, 1], 0) for _, pos in ball_xy_positions]
    ball_0_positions = ndarray(ball_0_positions).copy()
    ball_1_positions = ndarray(ball_1_positions).copy()
    ball_positions = [ball_0_positions, ball_1_positions]
    # Extract the initial position and velocity of each ball.
    init_positions = []
    init_angular_velocities = []
    for b in ball_positions:
        init_positions.append(b[0])
        # Estimate the angular velocity using the first 5 frames.
        # Dist = omega * time * radius.
        steps = 5
        offset = b[steps] - b[0]
        dist = np.linalg.norm(offset)
        omega_mag = dist / ball_radius / (dt * substeps * steps)
        if dist < 5e-3:
            w = [0, 0, 0]
        else:
            w = ndarray([-offset[1], offset[0], 0]) / dist * omega_mag
        init_angular_velocities.append(w)
    init_positions = ndarray(init_positions)
    init_angular_velocities = ndarray(init_angular_velocities)

    # Extract the initial camera information.
    camera_data = pickle.load(open(Path(root_path) / 'python/example/billiard_ball_calibration/experiment/intrinsic.data', 'rb'))
    R = camera_data['R']
    T = camera_data['T']
    alpha = camera_data['alpha']

    # Extract the reference pixels.
    reference_pixels = []
    for i in range(start_frame, end_frame):
        pxl = pickle.load(open(Path(root_path) / 'python/example/billiard_ball_calibration'
            / 'experiment' / '{:04d}_centroid.data'.format(i), 'rb'))
        pxl = pxl_to_cal(pxl)
        pxl[:, 0] -= img_width / 2
        pxl[:, 1] -= img_height / 2
        reference_pixels.append(pxl)
    reference_pixels = ndarray(reference_pixels)

    # Build the environment.
    env = BilliardBallEnv3d(folder, {
        'init_positions': init_positions,
        'init_angular_velocities': init_angular_velocities,
        'radius': ball_radius,
        'reference_positions': ball_positions,
        'substeps': substeps,
        'state_force_parameters': [3e2, 3e2, 0.2, 0.2],
        'loss_type': '2d',
        'R_init': R,
        'T_init': T,
        'alpha_init': alpha,
        'rpy': [0, 0, 0],
        't': [0, 0, 0],
        'a': 1,
        'reference_pixels': reference_pixels,
    })
    deformable = env.deformable()
    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Decision variables to optimize:
    # - stiffness and frictional coefficient of the contact model.
    # - theta and scale of the initial angular velocity of the balls.
    # - initial locations of the balls.
    def get_init_state(x):
        x = ndarray(x).copy().ravel()
        assert x.size == 19
        log_stiff0, coeff0, log_stiff1, coeff1 = x[:4]
        theta0, scale0, theta1, scale1 = x[4:8]
        pos0 = [x[8], x[9], 0]
        pos1 = [x[10], x[11], 0]
        rpy = x[12:15]
        t = x[15:18]
        a = x[18]
        stiff0 = 10 ** log_stiff0
        stiff1 = 10 ** log_stiff1
        c0, s0 = np.cos(theta0), np.sin(theta0)
        c1, s1 = np.cos(theta1), np.sin(theta1)
        w = ndarray([[c0 * scale0, s0 * scale0, 0],
            [c1 * scale1, s1 * scale1, 0]])
        e = BilliardBallEnv3d(folder, {
            'init_positions': [pos0, pos1],
            'init_angular_velocities': w,
            'radius': ball_radius,
            'reference_positions': ball_positions,
            'substeps': substeps,
            'state_force_parameters': [stiff0, stiff1, coeff0, coeff1],
            'loss_type': '2d',
            'R_init': R,
            'T_init': T,
            'alpha_init': alpha,
            'rpy': rpy,
            't': t,
            'a': a,
            'reference_pixels': reference_pixels,
        })
        dc_dx = np.zeros((6, 19))
        dc_dx[0, 8] = 1
        dc_dx[1, 9] = 1
        dc_dx[3, 10] = 1
        dc_dx[4, 11] = 1
        dw_dx = np.zeros((6, 19))
        dw_dx[0, 4] = scale0 * -s0
        dw_dx[0, 5] = c0
        dw_dx[1, 4] = scale0 * c0
        dw_dx[1, 5] = s0
        dw_dx[3, 6] = scale1 * -s1
        dw_dx[3, 7] = c1
        dw_dx[4, 6] = scale1 * c1
        dw_dx[4, 7] = s1
        dp_dx = np.zeros((4, 19))
        dp_dx[0, 0] = (10 ** log_stiff0) * np.log(10)
        dp_dx[1, 2] = (10 ** log_stiff1) * np.log(10)
        dp_dx[2, 1] = 1
        dp_dx[3, 3] = 1
        info = { 'env': e, 'q0': e.default_init_position(), 'v0': e.default_init_velocity(),
            'dc_dx': dc_dx, 'dw_dx': dw_dx, 'dp_dx': dp_dx }
        return info

    # Optimization.
    init_theta0 = np.arctan2(init_angular_velocities[0, 1], init_angular_velocities[0, 0])
    init_scale0 = np.linalg.norm(init_angular_velocities[0])
    init_theta1 = np.arctan2(init_angular_velocities[1, 1], init_angular_velocities[1, 0])
    init_scale1 = np.linalg.norm(init_angular_velocities[1])
    x_lower = ndarray([
        1.5, 0.3, 1.5, 0.3, init_theta0 - 0.05, init_scale0 * 2.0, init_theta1 - 0.05, init_scale1 * 2.0,
        init_positions[0, 0] - 0.04, init_positions[0, 1] - 0.04, init_positions[1, 0] - 0.04, init_positions[1, 1] - 0.04,
        -0.2, -0.2, -0.2, -0.1, -0.1, -0.1, 0.8,
    ])
    x_upper = ndarray([
        2.0, 0.7, 2.0, 0.7, init_theta0 + 0.05, init_scale0 * 4.0, init_theta1 + 0.05, init_scale1 * 4.0,
        init_positions[0, 0] + 0.04, init_positions[0, 1] + 0.04, init_positions[1, 0] + 0.04, init_positions[1, 1] + 0.04,
        0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 1.2
    ])
    bounds = scipy.optimize.Bounds(x_lower, x_upper)
    x_init = np.random.uniform(low=x_lower, high=x_upper)

    # Normalize the loss.
    random_guess_num = 4
    random_loss = []
    best_loss = np.inf
    best_x_init = None
    for _ in range(random_guess_num):
        x_guess = np.random.uniform(low=x_lower, high=x_upper)
        init_info = get_init_state(x_guess)
        e = init_info['env']
        v = init_info['v0']
        q = init_info['q0']
        loss, _ = e.simulate(dt, frame_num, pd_method, pd_opt, q, v, a0, f0, require_grad=False)
        print('loss:', loss)
        random_loss.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_x_init = np.copy(x_guess)
    loss_range = ndarray([0, np.mean(random_loss)])
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    x_init = best_x_init

    # Visualize the initial solution.
    if not (folder / 'init/0000.png').is_file():
        info = get_init_state(x_init)
        e_init = info['env']
        v_init = info['v0']
        q_init = info['q0']
        _, info = e_init.simulate(dt, frame_num, pd_method, pd_opt, q_init, v_init, a0, f0, require_grad=False, vis_folder='init',
            render_frame_skip=substeps)
        pickle.dump((x_init, info), open(folder / 'init/info.data', 'wb'))
        fig = plt.figure()
        ax = fig.add_subplot(211)
        q = info['q']
        traj = ndarray([np.mean(qi.reshape((2, -1, 3)), axis=1) for qi in q])
        ax.plot(traj[:, 0, 0], traj[:, 0, 1], color='tab:red', marker='+', label='ball_0_sim')
        ax.plot(traj[:, 1, 0], traj[:, 1, 1], color='tab:blue', marker='+', label='ball_1_sim')
        ax.plot(ball_0_positions[:, 0], ball_0_positions[:, 1], 'tab:red', label='ball_0_real')
        ax.plot(ball_1_positions[:, 0], ball_1_positions[:, 1], 'tab:blue', label='ball_1_real')
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('top down view')
        ax = fig.add_subplot(212)
        # The camera view.
        predicted_pixels = []
        for i in range(start_frame, end_frame):
            predicted_pixels.append(e_init.get_pixel_location(info['q'][int((i - start_frame) * substeps)]))
        predicted_pixels = ndarray(predicted_pixels)
        colors = ('tab:red', 'tab:blue')
        for i in range(predicted_pixels.shape[1]):
            px = predicted_pixels[:, i, 0]
            py = predicted_pixels[:, i, 1]
            ax.plot(px + img_width / 2, py + img_height / 2, color=colors[i], marker='+')
            rx = reference_pixels[:, i, 0]
            ry = reference_pixels[:, i, 1]
            ax.plot(rx + img_width / 2, ry + img_height / 2, color=colors[i])
        ax.set_xlim([0, img_width])
        ax.set_ylim([0, img_height])
        ax.set_aspect('equal')
        ax.set_title('camera view')
        plt.show()
        fig.savefig(folder / 'init/compare.png')

    data = { 'loss_range': loss_range }
    for method, opt in zip(methods, opts):
        data[method] = []
        def loss_and_grad(x):
            init_info = get_init_state(x)
            e = init_info['env']
            q = init_info['q0']
            v = init_info['v0']
            loss, grad, info = e.simulate(dt, frame_num, method, opt, q, v, a0, f0, require_grad=True)
            # We start from 3: because the first three state parameters are gravitiy.
            g = info['state_force_parameter_gradients'][3:] @ init_info['dp_dx'] \
                + e.backprop_init_velocities(grad[1]) @ init_info['dw_dx'] \
                + e.backprop_init_positions(grad[0]) @ init_info['dc_dx']
            c_g = info['grad_custom']
            g[12:15] += c_g['rpy']
            g[15:18] += c_g['t']
            g[18] += c_g['a']
            print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(g), info['forward_time'], info['backward_time']))

            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(g)
            single_data['x'] = np.copy(x)
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, ndarray(g).copy().ravel()

        # Use the two lines below to sanity check the gradients.
        # Note that you might need to fine tune the rel_tol in opt to make it work.
        # from py_diff_pd.common.grad_check import check_gradients
        # check_gradients(loss_and_grad, x_init, eps=1e-6)

        t0 = time.time()
        def callback(xk):
            print_info('Another iteration is finished.')
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-4, 'maxfun': 40, 'maxiter': 10 }, callback=callback)
        t1 = time.time()
        if not result.success:
            print_warning('Optimization is not successful. Using the last iteration results.')
            idx = np.argmin([d['loss'] for d in data[method]])
            print_warning('Using loss =', data[method][idx]['loss'])
            x_final = data[method][idx]['x']
        else:
            x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize the final results.
        if not (folder / method / '0000.png').is_file():
            info = get_init_state(x_final)
            e_init = info['env']
            v_init = info['v0']
            q_init = info['q0']
            _, info = e_init.simulate(dt, frame_num, method, opt, q_init, v_init, a0, f0, require_grad=False, vis_folder=method,
                render_frame_skip=substeps)
            pickle.dump((x_final, info), open(folder / method / 'info.data', 'wb'))
            fig = plt.figure()
            ax = fig.add_subplot(211)
            q = info['q']
            traj = ndarray([np.mean(qi.reshape((2, -1, 3)), axis=1) for qi in q])
            ax.plot(traj[:, 0, 0], traj[:, 0, 1], color='tab:red', marker='+', label='ball_0_sim')
            ax.plot(traj[:, 1, 0], traj[:, 1, 1], color='tab:blue', marker='+', label='ball_1_sim')
            ax.plot(ball_0_positions[:, 0], ball_0_positions[:, 1], 'tab:red', label='ball_0_real')
            ax.plot(ball_1_positions[:, 0], ball_1_positions[:, 1], 'tab:blue', label='ball_1_real')
            ax.set_aspect('equal')
            ax.legend()
            ax.set_title('top down view')
            ax = fig.add_subplot(212)
            # The camera view.
            predicted_pixels = []
            for i in range(start_frame, end_frame):
                predicted_pixels.append(e_init.get_pixel_location(info['q'][int((i - start_frame) * substeps)]))
            predicted_pixels = ndarray(predicted_pixels)
            colors = ('tab:red', 'tab:blue')
            for i in range(predicted_pixels.shape[1]):
                px = predicted_pixels[:, i, 0]
                py = predicted_pixels[:, i, 1]
                ax.plot(px + img_width / 2, py + img_height / 2, color=colors[i], marker='+')
                rx = reference_pixels[:, i, 0]
                ry = reference_pixels[:, i, 1]
                ax.plot(rx + img_width / 2, ry + img_height / 2, color=colors[i])
            ax.set_xlim([0, img_width])
            ax.set_ylim([0, img_height])
            ax.set_aspect('equal')
            ax.set_title('camera view')
            plt.show()
            fig.savefig(folder / method / 'compare.png')