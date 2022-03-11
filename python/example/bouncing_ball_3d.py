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
from py_diff_pd.env.bouncing_ball_env_3d import BouncingBallEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('bouncing_ball_3d')
    refinement = 8
    youngs_modulus = 2e6
    poissons_ratio = 0.4
    env = BouncingBallEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 4e-3
    frame_num = 125

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_gt = ndarray([0.0, 0.0, 0.07])
    v_gt = ndarray([1.5, 0.5, -1])
    q0 = env.default_init_position()
    q0 = (q0.reshape((-1, 3)) + q_gt).ravel()
    v0 = np.zeros(dofs)
    v0 = (v0.reshape((-1, 3)) + v_gt).ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groundtruth motion.
    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a0, f0, require_grad=False, vis_folder='groundtruth')

    # Optimization.
    # Decision variables: log(E), log(nu).
    x_lb = ndarray([np.log(1e6), np.log(0.2)])
    x_ub = ndarray([np.log(1e7), np.log(0.45)])
    # For this example, we intentionally pick an initial guess far from the ground truth so that the initial motion
    # is distinguishable enough from the ground truth motion.
    x_init = ndarray([np.log(1e7) - 0.01, np.log(0.45) - 0.01])
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    # Generate initial motion.
    E = np.exp(x_init[0])
    nu = np.exp(x_init[1])
    env_opt = BouncingBallEnv3d(seed, folder, { 'youngs_modulus': E, 'poissons_ratio': nu })
    env_opt.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a0, f0, require_grad=False, vis_folder='init')

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 16
    random_loss = []
    for _ in range(random_guess_num):
        x_rand = np.random.uniform(low=x_lb, high=x_ub)
        E = np.exp(x_rand[0])
        nu = np.exp(x_rand[1])
        env_opt = BouncingBallEnv3d(seed, folder, { 'youngs_modulus': E, 'poissons_ratio': nu })
        loss, _ = env_opt.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a0, f0, require_grad=False, vis_folder=None)
        print('E: {:3e}, nu: {:3f}, loss: {:3f}'.format(E, nu, loss))
        random_loss.append(loss)
    loss_range = ndarray([0, np.mean(random_loss)])
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    np.random.set_state(rand_state)

    data = { 'loss_range': loss_range }
    for method, opt in zip(reversed(methods), reversed(opts)):
        data[method] = []
        def loss_and_grad(x):
            E = np.exp(x[0])
            nu = np.exp(x[1])
            env_opt = BouncingBallEnv3d(seed, folder, { 'youngs_modulus': E,
                'poissons_ratio': nu })
            loss, _, info = env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=True, vis_folder=None)
            grad = info['material_parameter_gradients']
            grad = grad * np.exp(x)
            print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad), E, nu, info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad)
            single_data['E'] = E
            single_data['nu'] = nu
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, grad
        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-2, 'maxiter': 10 })
        t1 = time.time()
        print(result.success)
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        E = np.exp(x_final[0])
        nu = np.exp(x_final[1])
        env_opt = BouncingBallEnv3d(seed, folder, { 'refinement': refinement, 'youngs_modulus': E,
            'poissons_ratio': nu })
        env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=method)
