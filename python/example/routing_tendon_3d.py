import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle
from tqdm import tqdm

from py_diff_pd.common.common import ndarray, create_folder, rpy_to_rotation, rpy_to_rotation_gradient
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.routing_tendon_env_3d import RoutingTendonEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('routing_tendon_3d')
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    target = ndarray([0.2, 0.2, 0.45])
    refinement = 2
    muscle_cnt = 4
    muscle_ext = 4
    act_max = 2
    env = RoutingTendonEnv3d(seed, folder, {
        'muscle_cnt': muscle_cnt,
        'muscle_ext': muscle_ext,
        'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'target': target })
    deformable = env.deformable()

    # Optimization parameters.
    import multiprocessing
    cpu_cnt = multiprocessing.cpu_count()
    thread_ct = cpu_cnt - 1
    print_info('Detected {:d} CPUs. Using {} of them in this example'.format(cpu_cnt, thread_ct)) 
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('pd_eigen', 'newton_pcg', 'newton_cholesky')
    opts = (pd_opt, newton_opt, newton_opt)

    dt = 1e-2
    frame_num = 100

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]
    act_maps = env.act_maps()
    u_dofs = len(act_maps)
    assert u_dofs * (refinement ** 3) * muscle_ext == act_dofs

    def variable_to_act(x):
        act = np.zeros(act_dofs)
        for i, a in enumerate(act_maps):
            act[a] = x[i]
        return act
    def variable_to_act_gradient(x, grad_act):
        grad_u = np.zeros(u_dofs)
        for i, a in enumerate(act_maps):
            grad_u[i] = np.sum(grad_act[a])
        return grad_u

    # Optimization.
    x_lb = np.zeros(u_dofs)
    x_ub = np.ones(u_dofs) * act_max
    x_init = np.random.uniform(x_lb, x_ub)
    # Visualize initial guess.
    a_init = variable_to_act(x_init)
    print_info('Simulating and rendering initial solution. Please check out the {}/init folder'.format(folder))
    env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, [a_init for _ in range(frame_num)], f0, require_grad=False, vis_folder='init')
    print_info('Initial guess is ready. You can play it by opening {}/init.gif'.format(folder))

    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 16
    random_loss = []
    print_info('Randomly sample {} initial solutions to get a rough idea of the average performance'.format(random_guess_num))
    for _ in tqdm(range(random_guess_num)):
        x_rand = np.random.uniform(low=x_lb, high=x_ub)
        a = variable_to_act(x_rand)
        loss, _ = env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, [a for _ in range(frame_num)], f0,
            require_grad=False, vis_folder=None)
        random_loss.append(loss)
    loss_range = ndarray([0, np.mean(random_loss)])
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    np.random.set_state(rand_state)

    data = { 'loss_range': loss_range }
    method_display_names = { 'pd_eigen': 'DiffPD', 'newton_pcg': 'PCG', 'newton_cholesky': 'Cholesky' }
    for method, opt in zip(methods, opts):
        data[method] = []
        print_info('Optimizing with {}...'.format(method_display_names[method]))
        def loss_and_grad(x):
            a = variable_to_act(x)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, q0, v0, [a for _ in range(frame_num)], f0,
                require_grad=True, vis_folder=None)
            # Assemble the gradients.
            grad_a = 0
            for ga in grad[2]:
                grad_a += ga
            grad_x = variable_to_act_gradient(x, grad_a)
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
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method_display_names[method], t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        final_a = variable_to_act(x_final)
        print_info('Simulating and rendering final results from {}...'.format(method_display_names[method]))
        print_info('You can check out rendering results in {}/{}'.format(folder, method))
        env.simulate(dt, frame_num, method, opt, q0, v0, [final_a for _ in range(frame_num)], f0,
            require_grad=False, vis_folder=method)
        print_info('Results ready for review. You can play it by opening {}/{}.gif'.format(folder, method))