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
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.hopper_env_3d import HopperEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('hopper_3d')
    refinement = 2
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    leg_width = 2
    half_leg_height = 2
    waist_height = 2
    waist_width = 2
    thickness = 1

    env = HopperEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'leg_width': leg_width,
        'half_leg_height': half_leg_height,
        'waist_height': waist_height,
        'waist_width': waist_width,
        'thickness': thickness
    })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 2e-3
    frame_num = 200
    control_frame_num = 10
    assert frame_num % control_frame_num == 0

    # Compute the initial state.
    dofs = deformable.dofs()
    q0 = env.default_init_position()
    init_offset = ndarray([0, 0, 0.025])
    q0 = (q0.reshape((-1, 3)) + init_offset).ravel()
    v0 = env.default_init_velocity() * 0.5
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Compute actuation.
    control_frame = int(frame_num // control_frame_num)
    x_lb = np.zeros(2 * control_frame)
    x_ub = np.ones(2 * control_frame) * 1.25
    x_init = np.random.uniform(low=x_lb, high=x_ub) * 0.001 + (x_ub + x_lb) / 2.0
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    # Initial guess.
    def variable_to_act(x):
        x = np.copy(ndarray(x)).reshape((control_frame, 2))
        acts = []
        for u in x:
            frame_act = np.concatenate([
                np.ones(len(env.left_leg_left_fiber())) * u[0],
                np.ones(len(env.left_leg_right_fiber())) * (2 - u[0]),
                np.ones(len(env.right_leg_left_fiber())) * u[1],
                np.ones(len(env.right_leg_right_fiber())) * (2 - u[1]),
            ])
            acts += [np.copy(frame_act) for _ in range(control_frame_num)]
        return acts

    a0 = variable_to_act(x_init)
    env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, vis_folder='init')

    def variable_to_gradient(x, dl_dact):
        x = np.copy(ndarray(x)).reshape((control_frame, 2))
        grad = np.zeros(x.shape)
        for cf_idx, u in enumerate(x):
            for f in range(control_frame_num):
                f_idx = cf_idx * control_frame_num + f
                grad_act = dl_dact[f_idx]
                ll_size = len(env.left_leg_left_fiber())
                lr_size = len(env.left_leg_right_fiber())
                rl_size = len(env.right_leg_left_fiber())
                rr_size = len(env.right_leg_right_fiber())
                grad[cf_idx, 0] += np.sum(grad_act[:ll_size]) - np.sum(grad_act[ll_size:ll_size + lr_size])
                grad[cf_idx, 1] += np.sum(grad_act[-rr_size - rl_size:-rr_size]) - np.sum(grad_act[-rr_size:])
        return grad.ravel()

    # Sanity check.
    random_weight = np.random.normal(size=ndarray(a0).shape)
    def loss_and_grad(x):
        act = variable_to_act(x)
        loss = np.sum(ndarray(act) * random_weight)
        grad = variable_to_gradient(x, random_weight)
        return loss, grad
    check_gradients(loss_and_grad, x_init, verbose=True)

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 16
    random_loss = []
    for _ in range(random_guess_num):
        x_rand = np.random.uniform(low=x_lb, high=x_ub)
        act = variable_to_act(x_rand)
        loss, _ = env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, act, f0, require_grad=False, vis_folder=None)
        random_loss.append(loss)
    loss_range = ndarray([0, np.mean(random_loss)])
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    np.random.set_state(rand_state)

    # Optimization.
    data = { 'loss_range': loss_range }
    for method, opt in zip(methods, opts):
        data[method] = []
        def loss_and_grad(x):
            act = variable_to_act(x)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, q0, v0, act, f0, require_grad=True, vis_folder=None)
            dl_act = grad[2]
            grad = variable_to_gradient(x, dl_act)

            print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad), info['forward_time'], info['backward_time']))

            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad)
            single_data['x'] = np.copy(x)
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, np.copy(grad)

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-4 })
        t1 = time.time()
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        a_final = variable_to_act(x_final)
        env.simulate(dt, frame_num, method, opt, q0, v0, a_final, f0, require_grad=False, vis_folder=method)
