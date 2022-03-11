import sys
sys.path.append('../')

import os
from pathlib import Path
import time
from pathlib import Path
import scipy.optimize
import numpy as np

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.env.hopper_env_2d import HopperEnv2d

def test_collision_2d(verbose):
    seed = 42
    folder = Path('collision_2d')
    env = HopperEnv2d(seed, folder, { 'refinement': 1 })
    deformable = env.deformable()

    methods = ['newton_pcg', 'newton_cholesky', 'pd_eigen']
    opts = [{ 'max_newton_iter': 200, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-8, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 200, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-8, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 200, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-8, 'verbose': 0, 'thread_ct': 4,
            'use_bfgs': 1, 'bfgs_history_size': 10 }]
    if 'PARDISO_LIC_PATH' in os.environ:
        methods += ['newton_pardiso', 'pd_pardiso']
        opts.append(opts[1])
        opts.append(opts[2])

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = np.random.uniform(size=act_dofs)
    f0 = np.random.normal(scale=0.1, size=dofs)

    dt = 5e-3
    frame_num = 50

    # Compare forward.
    losses = {}
    qs = {}
    for method, opt in zip(methods, opts):
        loss, info = env.simulate(dt, frame_num, method, opt, q0, v0, [a0 for _ in range(frame_num)],
            [f0 for _ in range(frame_num)], require_grad=False, vis_folder=method if verbose else None)
        losses[method] = loss
        qs[method] = info['q']
        if verbose:
            os.system('eog {}.gif'.format(folder / method))

    eps = 1e-3
    atol = 1e-3
    rtol = 5e-2
    for method in methods:
        if not np.isclose(losses['newton_pcg'], losses[method]):
            if verbose:
                print_error('Losses are inconsistent between newton_pcg and {}'.format(method))
            return False
        for q, qm in zip(qs['newton_pcg'], qs[method]):
            if not np.allclose(q, qm, atol=atol, rtol=rtol):
                if verbose:
                    print_error('states are inconsistent between newton_pcg and {}'.format(method))
                return False

    def skip_var(dof):
        return env.is_dirichlet_dof(dof)

    for method, opt in zip(methods, opts):
        if verbose:
            print_info('Checking gradients in {} method.'.format(method))
        t0 = time.time()
        x0 = np.concatenate([q0, v0, a0, f0])

        def loss_and_grad(x):
            q_init = x[:dofs]
            v_init = x[dofs:2 * dofs]
            act = [x[2 * dofs:2 * dofs + act_dofs] for _ in range(frame_num)]
            f_ext = [x[2 * dofs + act_dofs:] for _ in range(frame_num)]
            loss, grad, _ = env.simulate(dt, frame_num, method, opt, q_init, v_init, act, f_ext, require_grad=True, vis_folder=None)

            grad_q, grad_v, grad_a, grad_f = grad
            grad = np.zeros(x.size)
            grad[:dofs] = grad_q
            grad[dofs:2 * dofs] = grad_v
            grad[2 * dofs:2 * dofs + act_dofs] = np.sum(ndarray(grad_a), axis=0)
            grad[2 * dofs + act_dofs:] = np.sum(ndarray(grad_f), axis=0)
            return loss, grad

        def loss(x):
            q_init = x[:dofs]
            v_init = x[dofs:2 * dofs]
            act = [x[2 * dofs:2 * dofs + act_dofs] for _ in range(frame_num)]
            f_ext = [x[2 * dofs + act_dofs:] for _ in range(frame_num)]
            loss, _ = env.simulate(dt, frame_num, method, opt, q_init, v_init, act, f_ext, require_grad=False, vis_folder=None)
            return loss

        if not check_gradients(loss_and_grad, x0, eps, rtol, atol, verbose, skip_var=skip_var, loss_only=loss):
            if verbose:
                print_error('Gradient check failed at {}'.format(method))
            return False
        t1 = time.time()
        print_info('Gradient check finished in {:3.3f}s.'.format(t1 - t0))

    return True

if __name__ == '__main__':
    verbose = True
    test_collision_2d(verbose)