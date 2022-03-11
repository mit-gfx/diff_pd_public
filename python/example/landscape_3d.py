import sys
sys.path.append('../')

from pathlib import Path
import pickle
import numpy as np

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info
from py_diff_pd.env.cantilever_env_3d import CantileverEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('landscape_3d')
    env = CantileverEnv3d(seed, folder, { 'refinement': 3 })
    deformable = env.deformable()

    opts = {}
    opts['semi_implicit'] = { 'thread_ct': 4 }
    opts['pd_eigen'] = { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
        'use_bfgs': 1, 'bfgs_history_size': 10 }

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = np.random.uniform(size=act_dofs)
    f0 = np.random.normal(scale=0.1, size=dofs) * 1e-3

    # Experiment 1: figure out the largest dt that semi_implicit solver supports.
    dts = [1e-4, 2e-4, 5e-4,]# 1e-3, 2e-3, 5e-3, 1e-2]
    frame_nums = [2000, 1000, 400,]# 200, 100, 40, 20]
    for dt, frame_num in zip(dts, frame_nums):
        loss, info = env.simulate(dt, frame_num, 'semi_implicit', opts['semi_implicit'], q0, v0,
            [a0 for _ in range(frame_num)], [f0 for _ in range(frame_num)], require_grad=False, vis_folder=None)
        print('dt: {:3.3e}, loss: {:3.3e}, time: {:3.3f}s'.format(dt, loss, info['forward_time']))

    dts = {}
    dts['semi_implicit'] = 5e-4
    dts['pd_eigen'] = 1e-2
    frame_nums = {}
    frame_nums['semi_implicit'] = 400
    frame_nums['pd_eigen'] = 20

    ss = np.linspace(-1, 1, 21)
    all_losses = { 'semi_implicit': [], 'pd_eigen': [] }
    all_grads = { 'semi_implicit': [], 'pd_eigen': [] }
    all_dqs = []
    # Try different directional gradients.
    for i in range(16):
        print_info('Generating {}/16 directional gradients...'.format(i))
        dq = np.random.uniform(low=-0.001, high=0.001, size=dofs)
        all_dqs.append(dq)
        for method in ['semi_implicit', 'pd_eigen']:
            dt = dts[method]
            frame_num = frame_nums[method]
            losses = []
            grads = []
            for s in ss:
                loss, grad, _ = env.simulate(dt, frame_num, method, opts[method], q0 + s * dq, v0,
                    [a0 for _ in range(frame_num)], [f0 for _ in range(frame_num)], require_grad=True, vis_folder=None)
                losses.append(loss)
                grads.append(grad)
            all_losses[method].append(losses)
            all_grads[method].append(grads)

        # Save data.
        pickle.dump((all_losses, all_grads, ss, all_dqs), open(folder / 'table.bin', 'wb'))