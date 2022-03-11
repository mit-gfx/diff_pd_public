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
from py_diff_pd.env.quadruped_env_3d import QuadrupedEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('quadruped_3d')
    refinement = 3
    act_max = 1.0
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    leg_z_length = 2
    body_x_length = 4
    body_y_length = 4
    body_z_length = 1
    env = QuadrupedEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'leg_z_length': leg_z_length,
        'body_x_length': body_x_length,
        'body_y_length': body_y_length,
        'body_z_length': body_z_length })
    deformable = env.deformable()
    leg_indices = env._leg_indices

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-2
    frame_num = 100

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    init_offset = ndarray([0, 0, 0.025])
    q0 = (q0.reshape((-1, 3)) + init_offset).ravel()
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    x_lb = ndarray([0.5, 0.5, 2 * np.pi / frame_num])
    x_ub = ndarray([1, 1, 2 * np.pi / 5])
    x_init = ndarray([0.5 * np.random.random() + 0.5, 0.5 * np.random.random() + 0.5,
        np.random.uniform(2 * np.pi / frame_num, 2 * np.pi / 5)]) * 0.01
    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    def variable_to_acts(x, return_jac):
        A_f, A_b, w = x
        jac = [np.ones((3, act_dofs)) for _ in range(frame_num)]
        a = [np.zeros(act_dofs) for _ in range(frame_num)]
        for i in range(frame_num):
            for key, indcs in leg_indices.items():
                if key[-1] == 'F':
                    for idx in indcs:
                        if key[0] == 'F':
                            a[i][idx] = act_max * (1 + A_f * np.sin(w * i)) / 2
                            jac[i][:, idx] = [np.sin(w * i), 0, A_f * i * np.cos(w * i)]
                        else:
                            a[i][idx] = act_max * (1 + A_b * np.sin(w * i)) / 2
                            jac[i][:, idx] = [0, np.sin(w * i), A_b * i * np.cos(w * i)]
                else:
                    for idx in indcs:
                        if key[0] =='F':
                            a[i][idx] =  act_max * (1 - A_f * np.sin(w * i)) / 2
                            jac[i][:, idx] = [-np.sin(w * i), 0, -A_f * i * np.cos(w * i)]
                        else:
                            a[i][idx] = act_max * (1 - A_b * np.sin(w * i)) / 2
                            jac[i][:, idx] = [0, -np.sin(w * i), -A_b * i * np.cos(w * i)]
        jac = [act_max * col / 2 for col in jac]
        if return_jac:
            return a, jac
        return a

    # Initial guess.
    a_init = variable_to_acts(x_init, False)
    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a_init, f0, require_grad=False, vis_folder='init')

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 16
    random_loss = []
    for _ in range(random_guess_num):
        x_rand = np.random.uniform(low=x_lb, high=x_ub)
        a = variable_to_acts(x_rand, False)
        loss, _ = env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a, f0, require_grad=False, vis_folder=None)
        random_loss.append(loss)
    loss_range = ndarray([0, np.mean(random_loss)])
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    np.random.set_state(rand_state)

    data = { 'loss_range': loss_range }
    # This example takes very long. As a result, I reverse the order so that I can see PD results first.
    for method, opt in zip(reversed(methods), reversed(opts)):
        data[method] = []
        def loss_and_grad(x):
            a, jac = variable_to_acts(x, True)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, q0, v0, a, f0, require_grad=True, vis_folder=None)
            # Assemble the gradients.
            act_grad = grad[2]
            grad = ndarray([jac[i].dot(np.transpose(act_grad[i])) for i in range(frame_num)])
            grad = np.sum(grad, axis=0)

            print('loss: {:8.3f}, |grad|: {:8.3f}, A_f: {:8.5f}, A_b: {:8.5f}, w: {:8.5f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad), x[0], x[1], x[2], info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad)
            single_data['x'] = np.copy(x)
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, np.copy(grad)

        # Use the two lines below to sanity check the gradients.
        # Note that you might need to fine tune the rel_tol in opt to make it work.
        # from py_diff_pd.common.grad_check import check_gradients
        # check_gradients(loss_and_grad, x_init, eps=1e-3)

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3, 'maxiter': 25 })
        t1 = time.time()
        print(result.success)
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        a_final = variable_to_acts(x_final, False)
        env.simulate(dt, frame_num, method, opt, q0, v0, a_final, f0, require_grad=False, vis_folder=method)
