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
from py_diff_pd.env.pingpong_env_2d import PingpongEnv2d

if __name__ == '__main__':
    seed = 42
    folder = Path('pingpong_2d')
    refinement = 4
    youngs_modulus = 1e5
    poissons_ratio = 0.45

    env = PingpongEnv2d(seed, folder, {
        'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
    })
    deformable = env.deformable()

    methods = ('newton_pcg', 'pd_eigen')
    opts = [
        { 'max_newton_iter': 200, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 200, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
            'use_bfgs': 1, 'bfgs_history_size': 10 }]

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = np.zeros(act_dofs)
    f0 = np.zeros(dofs)

    dt = 1e-2
    frame_num = 100
    a0 = [a0 for _ in range(frame_num)]
    f0 = [f0 for _ in range(frame_num)]

    def rotate(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = ndarray([[c, -s],
            [s, c]])
        return R

    def apply_transform(q, theta, t):
        R = rotate(theta)
        q = ndarray(q).reshape((-1, 2))
        com = np.mean(q, axis=0)
        return ((q - com) @ R.T + ndarray(t)).ravel()

    def apply_velocity(q, v_com, omega):
        q = ndarray(q).reshape((-1, 2))
        com = np.mean(q, axis=0)
        x, y = (q - com).T
        v = ndarray([-y, x]).T * omega + ndarray(v_com)
        return v.ravel()

    init_q_theta = 0
    init_q_com = ndarray([0.1, 0.5])
    init_q = apply_transform(q0, init_q_theta, init_q_com)
    init_v = apply_velocity(init_q, [1, -1], 10)
    target_q_com = ndarray([0.9, 0.75])
    target_q_theta = 0.1
    target_q = apply_transform(q0, target_q_theta, target_q_com)
    env.set_target_q(target_q)
    for method, opt in zip(methods, opts):
        loss, _, info = env.simulate(dt, frame_num, method, opt, init_q, init_v, a0, f0, require_grad=True, vis_folder=method)
        print('{} forward: {:3.3f}s; backward: {:3.3f}s'.format(method, info['forward_time'], info['backward_time']))

    # Optimization.
    pivot = ndarray([0.5, 0])
    x_lb = ndarray([-np.pi / 3, 0.1, -2.0, -np.pi])
    x_ub = ndarray([np.pi / 3, 1.0, -0.7, np.pi])
    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    x_init = np.random.uniform(x_lb, x_ub)
    def variable_to_initial_states(x):
        # x is 4D:
        # x[0]: pivot angle.
        # x[1], x[2]: linear velocity.
        # x[3]: omega.
        # init_q and target_q are fixed. What we optimize is the init_v and pivot angle.
        pivot_angle = x[0]
        v_com = x[1:3]
        omega = x[3]

        # Convert target_q and init_q into the body frame.
        R_world_to_body = rotate(-pivot_angle)
        target_q_body = ((target_q.reshape((-1, 2)) - pivot) @ R_world_to_body.T).ravel()
        init_q_body = ((init_q.reshape((-1, 2)) - pivot) @ R_world_to_body.T).ravel()

        # Convert init_v into the body frame.
        init_v = apply_velocity(init_q, v_com, omega)
        init_v_body = (init_v.reshape((-1, 2)) @ R_world_to_body.T).ravel()
        return init_q_body, init_v_body, target_q_body

    def variable_to_initial_states_gradient(x, grad_init_q_body, grad_init_v_body, grad_target_q_body):
        # TODO.
        pass

    for method, opt in zip(methods, opts):
        def loss_and_grad(x):
            init_q_body, init_v_body, target_q_body = variable_to_initial_states(x)
            env.set_target_q(target_q_body)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, init_q_body, init_v_body, a0, f0,
                require_grad=True, vis_folder=None)

            # Compute gradients.
            # See the loss function in PingpongEnv2d for the computatio of grad_target_q_body.
            grad_target_q_body = target_q_body - info['q'][-1]
            grad_init_q_body = grad[0]
            grad_init_v_body = grad[1]
            # Back-propagate through variable_to_initial_states.
            grad = variable_to_initial_states_gradient(x, grad_init_q_body, grad_init_v_body, grad_target_q_body)
            return loss, grad

        # BFGS begins here.
        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3 })
        t1 = time.time()
        assert result.success
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        init_q_body, init_v_body, target_q_body = variable_to_initial_states(x_final)
        env.set_target_q(target_q_body)
        # TODO: simulate the procedure in the world frame.
        env.simulate(dt, frame_num, method, opt, init_q_body, init_v_body, a0, f0, require_grad=False, vis_folder=method)