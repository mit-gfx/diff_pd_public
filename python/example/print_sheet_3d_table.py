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
    for s in (25, 50, 75, 100):
        seed = 42
        data_folder = Path('napkin_3d_{:d}x{:d}'.format(s, s))
        methods = ('newton_pcg', 'pd_eigen')
        dt = 2e-3
        frame_num = 125

        # The ratios [0.4, 0.8, 1.0, 1.6] correspond to the four scenarios in Fig. 7 of the paper.
        for ratio in [0.4, 0.8, 1.0, 1.6]:
            env = NapkinEnv3d(seed, data_folder, {
                'contact_ratio': ratio,
                'cell_nums': (s, s, 1),
                'spp': 1,
            })
            obs_center = ndarray([0, 0, 0]) # This is from NapkinEnv3d.
            obs_radius = 0.5                # This is from NapkinEnv3d.
            deformable = env.deformable()
            print('dofs:', deformable.dofs())
            f_idx = env.friction_node_idx()
            print_info('relative size of |C|:', len(f_idx) * 3 / deformable.dofs())
            folder = data_folder / 'ratio_{:3f}'.format(ratio)
            for method in methods:
                info = pickle.load(open(folder / '{}.data'.format(method), 'rb'))
                contact_idx = info['active_contact_indices']
                forward_time = info['forward_time_per_frame']
                contact_forward_time = []
                for i, t in enumerate(forward_time):
                    if len(contact_idx[i + 1]) > 0:
                        contact_forward_time.append(t)
                print('res:{}, ratio: {}, method: {}, time: {}'.format(s, ratio, method, np.mean(contact_forward_time)))