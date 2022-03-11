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
from py_diff_pd.env.torus_env_3d import TorusEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('torus_3d')
    youngs_modulus = 5e5
    poissons_ratio = 0.4
    act_stiffness = 2e5
    act_group_num = 8
    env = TorusEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'act_stiffness': act_stiffness,
        'act_group_num': act_group_num
    })
    deformable = env.deformable()

    # Optimization parameters.
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    thread_ct = 8
    opts = (
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct },
        { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct,
            'use_bfgs': 1, 'bfgs_history_size': 10 },
    )

    dt = 4e-3
    frame_num = 400
    control_frame_num = 20
    assert frame_num % control_frame_num == 0

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    init_offset = ndarray([0, 0, 0])
    q0 = (q0.reshape((-1, 3)) + init_offset).ravel()
    v0 = env.default_init_velocity()
    v0 = (v0.reshape((-1, 3)) + ndarray([0.25, 0.0, 0.0])).ravel()
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Compute actuation.
    control_frame = int(frame_num // control_frame_num)
    x_lb = np.zeros(act_group_num * control_frame)
    x_ub = np.ones(act_group_num * control_frame) * 2
    x_init = np.random.uniform(low=x_lb, high=x_ub)
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    act_groups = env.act_groups()
    def variable_to_act(x):
        x = ndarray(x.ravel()).reshape((control_frame, act_group_num))
        # Linear interpolation.
        x_aug = []
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                x_aug.append((1 - t) * x[c] + t * x[c_next])

        acts = []
        for x_aug_frame in x_aug:
            frame_act = np.zeros(act_dofs)
            for i, group in enumerate(act_groups):
                for j in group:
                    frame_act[j] = x_aug_frame[i]
            acts.append(frame_act)
        acts = ndarray(acts)
        return acts

    def variable_to_gradient(x, dl_dact):
        x = ndarray(x.ravel()).reshape((control_frame, act_group_num))
        # Linear interpolation.
        x_aug = []
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                x_aug.append((1 - t) * x[c] + t * x[c_next])

        grad_x_aug = np.zeros((frame_num, act_group_num))
        for k in range(frame_num):
            x_aug_frame = x_aug[k]
            grad_act = dl_dact[k]
            for i, group in enumerate(act_groups):
                for j in group:
                    grad_x_aug[k, i] += grad_act[j]

        # Backpropagate from grad_x_aug to grad.
        grad = np.zeros(x.shape)
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                grad[c] += (1 - t) * grad_x_aug[c * control_frame_num + i]
                grad[c_next] += t * grad_x_aug[c * control_frame_num + i]

        return grad.ravel()

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 16
    random_loss = []
    # Since this example is easily trapped in local minima, we pick the best random guesses as the initial state.
    x_init = None
    best_loss = np.inf
    for _ in range(random_guess_num):
        x_rand = np.random.uniform(low=x_lb, high=x_ub)
        act = variable_to_act(x_rand)
        loss, _ = env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, act, f0, require_grad=False, vis_folder=None)
        print('loss: {:3f}'.format(loss))
        random_loss.append(loss)
        if loss < best_loss:
            best_loss = loss
            x_init = x_rand
    loss_range = ndarray([0, np.mean(random_loss)])
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    np.random.set_state(rand_state)

    # Initial state after selecting the best from random guesses.
    a0 = variable_to_act(x_init)
    '''
    # Sanity check.
    random_weight = np.random.normal(size=ndarray(a0).shape)
    def loss_and_grad(x):
        act = variable_to_act(x)
        loss = np.sum(ndarray(act) * random_weight)
        grad = variable_to_gradient(x, random_weight)
        return loss, grad
    check_gradients(loss_and_grad, x_init, verbose=True)
    '''
    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a0, f0, require_grad=False, vis_folder='init')

    # Optimization.
    data = { 'loss_range': loss_range }
    for method, opt in zip(reversed(methods), reversed(opts)):
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
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3, 'maxiter': 10 })
        t1 = time.time()
        print(result.success)
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        a_final = variable_to_act(x_final)
        env.simulate(dt, frame_num, method, opt, q0, v0, a_final, f0, require_grad=False, vis_folder=method)