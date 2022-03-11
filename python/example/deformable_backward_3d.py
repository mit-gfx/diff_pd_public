import sys
sys.path.append('../')

import os
from pathlib import Path
import time
from pathlib import Path
import scipy.optimize
import numpy as np
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, PrettyTabular
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.env.cantilever_env_3d import CantileverEnv3d

def test_deformable_backward_3d(verbose):
    seed = 42
    folder = Path('deformable_backward_3d')
    env = CantileverEnv3d(seed, folder, { 'refinement': 1, 'youngs_modulus': 1e4, 'poissons_ratio': 0.4 })
    deformable = env.deformable()

    methods = ['newton_pcg', 'newton_cholesky', 'pd_eigen']
    opts = [{ 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-12, 'rel_tol': 1e-12, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-12, 'rel_tol': 1e-12, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-12, 'rel_tol': 1e-12, 'verbose': 0, 'thread_ct': 4,
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

    rtols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    losses = {}
    grads = {}
    for method in methods:
        losses[method] = []
        grads[method] = []

    dt = 0.01
    frame_num = 5
    for method, opt in zip(methods, opts):
        if verbose:
            print_info('method: {}'.format(method))
            tabular = PrettyTabular({
                'rel_tol': '{:3.3e}',
                'loss': '{:3.3f}',
                '|grad|': '{:3.3f}'
            })
            print_info(tabular.head_string())

        for rtol in rtols:
            opt['rel_tol'] = rtol
            loss, grad, _ = env.simulate(dt, frame_num, method, opt, q0, v0, [a0 for _ in range(frame_num)],
                [f0 for _ in range(frame_num)], require_grad=True, vis_folder=None)
            grad_q, grad_v, grad_a, grad_f = grad
            grad = np.zeros(q0.size + v0.size + a0.size + f0.size)
            grad[:dofs] = grad_q
            grad[dofs:2 * dofs] = grad_v
            grad[2 * dofs:2 * dofs + act_dofs] = np.sum(ndarray(grad_a), axis=0)
            grad[2 * dofs + act_dofs:] = np.sum(ndarray(grad_f), axis=0)
            grad_norm = np.linalg.norm(grad)
            if verbose:
                print(tabular.row_string({
                    'rel_tol': rtol,
                    'loss': loss,
                    '|grad|': grad_norm
                }))
            losses[method].append(loss)
            grads[method].append(grad_norm)

    pickle.dump((rtols, losses, grads), open(folder / 'table.bin', 'wb'))
    # Compare table.bin to table_master.bin.
    rtols_master, losses_master, grads_master = pickle.load(open(folder / 'table_master.bin', 'rb'))
    def compare_list(l1, l2):
        if len(l1) != len(l2): return False
        return np.allclose(l1, l2)
    if not compare_list(rtols, rtols_master):
        if verbose:
            print_error('rtols and rtols_master are different.')
        return False
    for method in methods:
        if not compare_list(losses[method], losses_master[method]):
            if verbose:
                print_error('losses[{}] and losses_master[{}] are different.'.format(method, method))
            return False
        if not compare_list(grads[method], grads_master[method]):
            if verbose:
                print_error('grads[{}] and grads_master[{}] are different.'.format(method, method))
            return False

    if verbose:
        # Plot loss and grad vs rtol.
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 5))
        ax_fb = fig.add_subplot(121)
        ax_f = fig.add_subplot(122)
        titles = ['loss', '|grad|']
        for title, ax, y in zip(titles, (ax_fb, ax_f), (losses, grads)):
            ax.set_xlabel('relative error')
            ax.set_ylabel('magnitude (/)')
            ax.set_xscale('log')
            ax.set_xlim(rtols[0], rtols[-1])
            for method in methods:
                ax.plot(rtols, y[method], label=method)
            ax.grid(True)
            ax.legend()
            ax.set_title(title)

        fig.savefig(folder / 'deformable_backward_3d_rtol.pdf')
        fig.savefig(folder / 'deformable_backward_3d_rtol.png')
        plt.show()

    # Check gradients.
    eps = 1e-4
    atol = 1e-3
    rtol = 5e-2
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
    test_deformable_backward_3d(verbose)