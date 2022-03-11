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
from py_diff_pd.env.hopper_env_2d import HopperEnv2d

if __name__ == '__main__':
    seed = 42
    folder = Path('hopper_2d')
    refinement = 4
    act_max = 2.0
    youngs_modulus = 1e6
    poissons_ratio = 0.49

    env = HopperEnv2d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio})
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 4
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 20, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 1, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-2
    frame_num = 25

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    print(act_dofs)
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    x_init = np.random.uniform(low=0.0, high=act_max, size=2*frame_num)
    x_lb = np.zeros(2*frame_num)
    x_ub = act_max * np.ones(2*frame_num)
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    def variable_to_states(x):
        actuations = np.ones(act_dofs*frame_num)

        for i in range(len(x)):
            musc_len = int(np.floor(act_dofs / 2))
            actuations[i*musc_len:(i+1)*musc_len] *= x[i]
        a = [actuations[i*act_dofs:(i+1)*act_dofs] for i in range(frame_num)]

        return a

    def variable_to_gradient(dl_dai):
        grad = np.zeros(2*frame_num)

        for i in range(frame_num):
            musc_len = int(np.floor(act_dofs / 2))
            grad[2*i] = np.sum(dl_dai[i][:musc_len])
            grad[2*i + 1] = np.sum(dl_dai[i][musc_len:act_dofs])

        return grad

    a0 = variable_to_states(x_init)
    env.simulate(dt, frame_num, methods[1], opts[1], q0, v0, a0, f0, require_grad=False, vis_folder='init')

    data = {}
    for method, opt in zip(methods, opts):
        data[method] = []
        def loss_and_grad(x):
            a = variable_to_states(x)

            loss, grad, info = env.simulate(dt, frame_num, method, opt, q0, v0, a, f0, require_grad=True, vis_folder=None)
            dl_act = grad[2]
            grad = variable_to_gradient(dl_act)

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

        print(result.success)
        print_info("Optimization with {} finished in {} seconds.".format(method, t1 - t0))
        x_final = result.x
        a_final = variable_to_states(x_final)
        env.simulate(dt, frame_num, method, opt, q0, v0, a_final, f0, require_grad=False, vis_folder=method)

        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

    #Test if single hop sequence functions reasonably well for two cycles. If not loss is too high
    for i in range(frame_num):
        a_final.append(a_final[i])

    frame_num *= 2
    f0 = [np.zeros(dofs) for _ in range(frame_num)]
    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a_final, f0, require_grad=False, vis_folder='final')
