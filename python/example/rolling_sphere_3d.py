import sys
sys.path.append('../')

import os
from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder, rpy_to_rotation, rpy_to_rotation_gradient
from py_diff_pd.common.common import print_info, print_ok, print_error, PrettyTabular
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.rolling_sphere_env_3d import RollingSphereEnv3d
from py_diff_pd.common.display import export_mp4

def test_rolling_sphere(verbose):
    seed = 42
    folder = Path('rolling_sphere_3d')
    refinement = 10
    youngs_modulus = 2e6
    poissons_ratio = 0.4
    env = RollingSphereEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    # Setting thread number.
    thread_cts = [2, 4, 8]

    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen', 'pd_no_acc')
    opts = ({ 'max_newton_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
            'use_bfgs': 1, 'bfgs_history_size': 10 },
        { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
            'use_bfgs': 1, 'bfgs_history_size': 10, 'use_acc': 0 })

    dt = 5e-3
    frame_num = 100

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position() + np.random.normal(scale=0.001, size=dofs)
    radius = env.radius()
    pivot = ndarray([radius, radius, 0])
    omega = ndarray([0, 10.0, 0])
    omega_x, omega_y, omega_z = omega
    omega_skewed = ndarray([
        [0, -omega_z, omega_y],
        [omega_z, 0, -omega_x],
        [-omega_y, omega_x, 0]
    ])
    v0 = (q0.reshape((-1, 3)) @ -omega_skewed).ravel()
    a0 = np.zeros(act_dofs)
    f0 = np.zeros(dofs)

    # Visualization.
    if verbose:
        for method, opt in zip(methods, opts):
            _, _, info = env.simulate(dt, frame_num, 'pd_eigen' if method == 'pd_no_acc' else method,
                opt, q0, v0, [a0 for _ in range(frame_num)],
            [f0 for _ in range(frame_num)], require_grad=True, vis_folder=method)
            print('{}: forward: {:3.3f}s; backward: {:3.3f}s'.format(method, info['forward_time'], info['backward_time']))
            export_mp4(folder / method, '{}.mp4'.format(str(folder / method)), fps=12)

    # Benchmark time.
    print('Reporting time cost. DoFs: {:d}, Contact DoFs: {:d}, frames: {:d}, dt: {:3.3e}'.format(dofs,
        env.contact_dofs(), frame_num, dt))
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
                loss, grad, info = env.simulate(dt, frame_num, 'pd_eigen' if method == 'pd_no_acc' else method,
                    opt, q0, v0, [a0 for _ in range(frame_num)],
                    [f0 for _ in range(frame_num)], require_grad=True, vis_folder=None)
                grad_q, grad_v, grad_a, grad_f = grad
                grad = np.zeros(q0.size + v0.size + a0.size + f0.size)
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
    test_rolling_sphere(verbose)