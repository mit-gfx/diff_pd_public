import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder, rpy_to_rotation, rpy_to_rotation_gradient
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.bunny_env_3d import BunnyEnv3d

def apply_transform(q, R, t):
    q = ndarray(q).reshape((-1, 3))
    com = np.mean(q, axis=0)
    return ((q - com) @ R.T + t).ravel()

if __name__ == '__main__':
    seed = 42
    folder = Path('bunny_3d')
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    target_com = ndarray([0.15, 0.15, 0.15])
    env = BunnyEnv3d(seed, folder, {
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'target_com': target_com,
        'mesh_type': 'hex' })   # Replace 'hex' with 'tet' if you want to try out the tet meshes.
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-3
    frame_num = 100

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    def variable_to_initial_states(x):
        init_rpy = x[:3]
        init_R = rpy_to_rotation(init_rpy)
        init_com_q = x[3:6]
        init_com_v = x[6:9]
        init_q = apply_transform(q0, init_R, init_com_q)
        init_v = (v0.reshape((-1, 3)) + init_com_v).ravel()
        return np.copy(init_q), np.copy(init_v)
    def variable_to_initial_states_gradient(x, grad_init_q, grad_init_v):
        grad = np.zeros(x.size)
        # init_rpy:
        offset = q0.reshape((-1, 3)) - np.mean(q0.reshape((-1, 3)), axis=0)
        # init_q = (offset @ R.T + init_com_q).ravel()
        rpy = x[:3]
        dR_dr, dR_dp, dR_dy = rpy_to_rotation_gradient(rpy)
        grad[0] = (offset @ dR_dr.T).ravel().dot(grad_init_q)
        grad[1] = (offset @ dR_dp.T).ravel().dot(grad_init_q)
        grad[2] = (offset @ dR_dy.T).ravel().dot(grad_init_q)
        # init_com_q:
        grad[3:6] = np.sum(grad_init_q.reshape((-1, 3)), axis=0)
        # init_com_v:
        grad[6:9] = np.sum(grad_init_v.reshape((-1, 3)), axis=0)
        return grad

    # Optimization.
    # Variables to be optimized:
    # init_rpy (3D), init_com_q (3D), init_com_v (3D).
    x_lb = ndarray([-np.pi / 3, -np.pi / 3, -np.pi / 3, -0.01, -0.01, 0.19, -1.5, -1.5, -5])
    x_ub = ndarray([np.pi / 3, np.pi / 3, np.pi / 3, 0.01, 0.01, 0.21, 1.5, 1.5, -3])
    x_init = np.random.uniform(x_lb, x_ub)
    # Visualize initial guess.
    init_q, init_v = variable_to_initial_states(x_init)
    env.simulate(dt, frame_num, methods[0], opts[0], init_q, init_v, a0, f0, require_grad=False, vis_folder='init')
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 16
    random_loss = []
    for _ in range(random_guess_num):
        x_rand = np.random.uniform(low=x_lb, high=x_ub)
        init_q, init_v = variable_to_initial_states(x_rand)
        loss, _ = env.simulate(dt, frame_num, methods[2], opts[2], init_q, init_v, a0, f0, require_grad=False, vis_folder=None)
        print('loss: {:3f}'.format(loss))
        random_loss.append(loss)
    loss_range = ndarray([0, np.mean(random_loss)])
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    np.random.set_state(rand_state)

    data = { 'loss_range': loss_range }
    for method, opt in zip(methods, opts):
        data[method] = []
        def loss_and_grad(x):
            init_q, init_v = variable_to_initial_states(x)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, init_q, init_v, a0, f0, require_grad=True, vis_folder=None)
            # Assemble the gradients.
            grad_init_q = grad[0]
            grad_init_v = grad[1]
            grad_x = variable_to_initial_states_gradient(x, grad_init_q, grad_init_v)
            print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad_x), info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad_x)
            single_data['x'] = np.copy(x)
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, np.copy(grad_x)

        # Use the two lines below to sanity check the gradients.
        # Note that you might need to fine tune the rel_tol in opt to make it work.
        # from py_diff_pd.common.grad_check import check_gradients
        # check_gradients(loss_and_grad, x_init, eps=1e-6)

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3 })
        t1 = time.time()
        assert result.success
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        final_q, final_v = variable_to_initial_states(x_final)
        env.simulate(dt, frame_num, method, opt, final_q, final_v, a0, f0, require_grad=False, vis_folder=method)