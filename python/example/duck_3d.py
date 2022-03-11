import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.env.duck_env_3d import DuckEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    folder = Path('duck_3d')
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
    methods = ('pd_eigen', 'newton_pcg', 'newton_cholesky')
    opts = (pd_opt, newton_opt, newton_opt)

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

    def variable_to_env_gradient(x, grad_p):
        grad = np.zeros(x.size)
        assert len(x) == 3
        assert len(grad_p) == 9
        grad_kn = grad_p[3]
        grad_kf = grad_p[4]
        grad_mu = grad_p[5]
        grad[0] = 10 ** x[0] * np.log(10) * grad_kn
        grad[1] = grad_kf
        grad[2] = 10 ** x[1] * np.log(10) * grad_mu
        return grad

    # Optimization.
    # Variables to be optimized:
    x_lb = ndarray([2, 0, 3])
    x_ub = ndarray([4, 1, 5])
    # Ground truth was generated using [3, 0.1, 4].
    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    loss_range = ndarray([0, 1])
    x_init = np.random.uniform(low=x_lb, high=x_ub)

    # Normalize the loss.
    random_guess_num = 4
    if random_guess_num > 0:
        rand_state = np.random.get_state()
        random_loss = []
        best_loss = np.inf
        for _ in range(random_guess_num):
            x_rand = np.random.uniform(low=x_lb, high=x_ub)
            init_env = variable_to_env(x_rand)
            loss, _ = init_env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, vis_folder=None)
            print('loss: {:3f}'.format(loss), 'x:', x_rand)
            random_loss.append(loss)
            if loss < best_loss:
                best_loss = loss
                x_init = x_rand
        loss_range = ndarray([0, np.mean(random_loss)])
        print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
        np.random.set_state(rand_state)

    # Visualize results.
    init_env = variable_to_env(x_init)
    init_env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, vis_folder='init', render_frame_skip=10)

    data = { 'loss_range': loss_range }
    for method, opt in zip(methods, opts):
        data[method] = []
        def loss_and_grad(x):
            env = variable_to_env(x)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=True, vis_folder=None)
            # Assemble the gradients.
            grad_x = variable_to_env_gradient(x, info['state_force_parameter_gradients'])
            print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad_x), info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad_x)
            single_data['x'] = np.copy(x)
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))
            return loss, np.copy(grad_x)

        # Use the two lines below to sanity check the gradients.
        # Note that you might need to fine tune the rel_tol in opt to make it work.
        # from py_diff_pd.common.grad_check import check_gradients
        # check_gradients(loss_and_grad, x_init, eps=1e-6)

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-4, 'maxiter': 5 })
        t1 = time.time()
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        final_env = variable_to_env(x_final)
        final_env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=method, render_frame_skip=10)
