import sys
sys.path.append('../')

import os
import subprocess
from pathlib import Path
import time
from pathlib import Path
import pickle
import scipy.optimize
import numpy as np

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, PrettyTabular
from py_diff_pd.env.armadillo_env_3d import ArmadilloEnv3d

def test_armadillo_3d(verbose):
    seed = 42
    folder = Path('armadillo_3d')
    env = ArmadilloEnv3d(seed, folder, {
        'youngs_modulus': 5e5,
        'init_rotate_angle': 0,
        'state_force_parameters': [0, 0, 0],    # No gravity.
        'spp': 4
    })
    deformable = env.deformable()

    # Setting thread number.
    thread_cts = [6,]
    methods = ('pd_eigen_pcg', 'newton_cholesky')
    opts = ({ 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
                'use_bfgs': 1, 'bfgs_history_size': 10 },
            { 'max_newton_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 },)

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = np.zeros(act_dofs)
    dt = 3e-2
    frame_num = 30
    f0 = [np.zeros(dofs).reshape((-1, 3)) for _ in range(frame_num)]
    for t in range(frame_num):
        f_min_x = ndarray([np.cos(-np.pi / 4 + t / frame_num * np.pi / 2),
            np.sin(-np.pi / 4 + t / frame_num * np.pi / 2), 0]) * t / frame_num
        for i in env.min_x_nodes():
            f0[t][i] = f_min_x
        for i in env.max_x_nodes():
            f0[t][i] = -f_min_x
    f0 = [fi.ravel() for fi in f0]

    if verbose:
        for method, opt in zip(methods, opts):
            env.simulate(dt, frame_num, 'pd_eigen' if method == 'pd_no_bfgs' else method,
                opt, q0, v0, [a0 for _ in range(frame_num)], f0, require_grad=False, vis_folder=method)

    # Benchmark time.
    print('Reporting time cost. DoFs: {:d}, frames: {:d}, dt: {:3.3e}'.format(dofs, frame_num, dt))
    rel_tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    forward_backward_times = {}
    forward_times = {}
    backward_times = {}
    losses = {}
    grads = {}
    for method in methods:
        for thread_ct in thread_cts:
            meth_thread_num = '{}_{}threads'.format(method, thread_ct)
            forward_backward_times[meth_thread_num] = []
            forward_times[meth_thread_num] = []
            backward_times[meth_thread_num] = []
            losses[meth_thread_num] = []
            grads[meth_thread_num] = []

    for rel_tol in rel_tols:
        print_info('rel_tol: {:3.3e}'.format(rel_tol))
        tabular = PrettyTabular({
            'method': '{:^30s}',
            'forward and backward (s)': '{:3.3f}',
            'forward only (s)': '{:3.3f}',
            'loss': '{:3.3f}',
            '|grad|': '{:3.3f}'
        })
        print_info(tabular.head_string())

        for method, opt in zip(methods, opts):
            opt['rel_tol'] = rel_tol
            for thread_ct in thread_cts:
                opt['thread_ct'] = thread_ct
                meth_thread_num = '{}_{}threads'.format(method, thread_ct)
                loss, grad, info = env.simulate(dt, frame_num, 'pd_eigen' if method == 'pd_no_bfgs' else method,
                    opt, q0, v0, [a0 for _ in range(frame_num)], f0, require_grad=True, vis_folder=None)
                grad_q, grad_v, grad_a, grad_f = grad
                grad = np.zeros(dofs + dofs + act_dofs + dofs)
                grad[:dofs] = grad_q
                grad[dofs:2 * dofs] = grad_v
                grad[2 * dofs:2 * dofs + act_dofs] = np.sum(ndarray(grad_a), axis=0)
                grad[2 * dofs + act_dofs:] = np.sum(ndarray(grad_f), axis=0)
                l, g, forward_time, backward_time = loss, grad, info['forward_time'], info['backward_time']
                print(tabular.row_string({
                    'method': meth_thread_num,
                    'forward and backward (s)': forward_time + backward_time,
                    'forward only (s)': forward_time,
                    'loss': l,
                    '|grad|': np.linalg.norm(g) }))
                forward_backward_times[meth_thread_num].append(forward_time + backward_time)
                forward_times[meth_thread_num].append(forward_time)
                backward_times[meth_thread_num].append(backward_time)
                losses[meth_thread_num].append(l)
                grads[meth_thread_num].append(g)
        pickle.dump((rel_tols, forward_times, backward_times, losses, grads), open(folder / 'table.bin', 'wb'))

if __name__ == '__main__':
    verbose = True
    test_armadillo_3d(verbose)
