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
from py_diff_pd.env.napkin_env_3d import NapkinEnv3d

if __name__ == '__main__':
    seed = 42
    parent_folder = Path('napkin_3d_50x50')
    create_folder(parent_folder, exist_ok=True)
    methods = ('newton_pcg', 'pd_eigen')
    dt = 2e-3
    frame_num = 125
    thread_ct = 8
    opts = [
        { 'max_newton_iter': 200, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct },
        { 'max_pd_iter': 200, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
            'use_bfgs': 1, 'use_sparse': 1, 'bfgs_history_size': 10 }]
    for ratio in [0.4, 0.8, 1.0, 1.6]:
        folder = parent_folder / 'ratio_{:3f}'.format(ratio)
        env = NapkinEnv3d(seed, folder, {
            'contact_ratio': ratio,
            'cell_nums': (50, 50, 1),
            'spp': 1,
        })
        deformable = env.deformable()

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = env.default_init_position()
        v0 = env.default_init_velocity()
        a0 = np.zeros(act_dofs)
        f0 = np.zeros(dofs)

        a0 = [a0 for _ in range(frame_num)]
        f0 = [f0 for _ in range(frame_num)]

        for method, opt in zip(methods, opts):
            loss, info = env.simulate(dt, frame_num, method, opt, np.copy(q0), np.copy(v0), a0, f0, require_grad=False,
                vis_folder=None)
            print('{} forward: {:3.3f}s'.format(method, info['forward_time']))
            pickle.dump(info, open(folder / '{}.data'.format(method), 'wb'))
