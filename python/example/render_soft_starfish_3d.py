import sys
sys.path.append('../')
import os

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle
import matplotlib.pyplot as plt

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.common.display import export_gif
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.soft_starfish_env_3d import SoftStarfishEnv3d
from py_diff_pd.common.project_path import root_path

def load_csv_data(csv_name):
    csv_name = Path(csv_name)
    data = {}
    with open(csv_name, 'r') as f:
        lines = f.readlines()
    # Line 0 is the header.
    data['time'] = []
    data['dl'] = []
    data['M1'] = []
    data['M2'] = []
    data['M3'] = []
    data['M4'] = []

    init_dl = None
    processed_begin_area = False
    for l in lines[1:]:
        l = l.strip()
        if l == '': continue
        item = [float(v) for v in l.split(',') if v != '']
        assert len(item) == 10
        # Skip if dl is NaN.
        if np.isnan(item[1]):
            continue
        # Also, I notice that at the beginning dl tends to stay at the same location for a while.
        # Skip those data too.
        if not processed_begin_area:
            if init_dl is None:
                init_dl = item[1]
                continue
            elif item[1] == init_dl:
                continue
            else:
                processed_begin_area = True

        t, dl, m1x, m1y, m2x, m2y, m3x, m3y, m4x, m4y = item
        data['time'].append(t)
        data['dl'].append(dl)
        data['M1'].append((m1x, m1y))
        data['M2'].append((m2x, m2y))
        data['M3'].append((m3x, m3y))
        data['M4'].append((m4x, m4y))
    # Normalize data.
    t = ndarray(data['time'])
    t -= t[0]
    t /= 1000
    data['time'] = t    # Now t is in the unit of seconds.
    dt = t[1:] - t[:-1]
    assert np.max(dt) - np.min(dt) < 1e-4 and np.abs(np.mean(dt) - 1 / 60) < 1e-4
    data['dt'] = np.mean(dt)
    dl = ndarray(data['dl'])
    dl /= 1000
    data['dl'] = dl     # Now dl is in the unit of meters.
    for i in range(1, 5):
        name = 'M{:d}'.format(i)
        mi_pos = ndarray(data[name])
        mi_pos /= 1000  # Now mi_pos is in the unit of meters.
        mi_pos -= mi_pos[0]
        # Convert the coordinates:
        # x -> x
        # y -> -z.
        data[name + '_rel_x'] = mi_pos[:, 0]
        data[name + '_rel_z'] = -mi_pos[:, 1]
    del data['M1']
    del data['M2']
    del data['M3']
    del data['M4']

    info = {}
    # Compute the velocity of each marker.
    t = data['time']
    for i in range(1, 5):
        name = 'M{:d}'.format(i)
        rel_x = data[name + '_rel_x']
        rel_z = data[name + '_rel_z']
        max_x_vel = np.max(np.abs(rel_x[1:] - rel_x[:-1])) / dt
        max_z_vel = np.max(np.abs(rel_z[1:] - rel_z[:-1])) / dt
        info[name + '_max_vel'] = np.max([max_x_vel, max_z_vel])
    return data, info

def load_latest_data(folder, name_prefix):
    cnt = 0
    while True:
        data_file_name = folder / '{}_{:04d}.data'.format(name_prefix, cnt)
        if not os.path.exists(data_file_name):
            cnt -= 1
            break
        cnt += 1
    data_file_name = folder / '{}_{:04d}.data'.format(name_prefix, cnt)
    print_info('Loading data from {}.'.format(data_file_name))
    return pickle.load(open(data_file_name, 'rb'))

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    round_iters = 3
    spp = 128 #CHANGE THIS WHEN RENDERING FOR REAL
    for i in range(round_iters):
        round_iter = i + 1
        print("Rendering Videos for Round {}.".format(round_iter))
        create_folder('render_soft_starfish_3d/round{:d}'.format(round_iter), exist_ok=True)
        folder = Path('render_soft_starfish_3d/round{:d}'.format(round_iter))
        last_folder = Path('soft_starfish_3d/round{:d}'.format(round_iter - 1))
        measurement_data, _ = load_csv_data(
            Path(root_path) / 'python/example/soft_starfish_3d/data_horizontal_cyclic{:d}.csv'.format(round_iter))
        if round_iter == 1:
            max_vel = np.inf
        else:
            _, info = load_csv_data(
                Path(root_path) / 'python/example/soft_starfish_3d/data_horizontal_cyclic{:d}.csv'.format(round_iter - 1))
            max_vel = np.max([info['M{:d}_max_vel'.format(i)] for i in range(1, 5)]) * 2
        print('Maximum allowable velocity:', max_vel)

        youngs_modulus = 5e5
        poissons_ratio = 0.4
        act_stiffness = 2e6
        substep = 5
        env = SoftStarfishEnv3d(seed, folder, {
            'youngs_modulus': youngs_modulus,
            'poissons_ratio': poissons_ratio,
            'act_stiffness': act_stiffness,
            'y_actuator': False,
            'z_actuator': True,
            'fix_center_x': False,
            'fix_center_y': True,
            'fix_center_z': True,
            'use_stepwise_loss': True,
            'data': measurement_data,
            'substep': substep,
            'spp': spp
        })
        deformable = env.deformable()

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = env.default_init_position()
        v0 = np.zeros(dofs)
        dt = measurement_data['dt'] / substep

        # Optimization parameters.
        newton_method = 'newton_pcg'
        pd_method = 'pd_eigen'
        thread_ct = 6
        newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4,
            'verbose': 0, 'thread_ct': thread_ct }
        pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4,
            'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 }

        ###########################################################################
        # System identification
        ###########################################################################
        # Conmpute actuation signals.
        actuator_signals = []
        frame_num = 60 * substep
        f0 = np.zeros(dofs)
        f0 = [f0 for _ in range(frame_num)]
        for i in range(frame_num):
            actuator_signal = 1 - np.ones(act_dofs) * measurement_data['dl'][int(i // substep)] / env.full_tendon_length()
            actuator_signals.append(actuator_signal)

        x_final = load_latest_data(folder, 'sys_id')[-1][0]

        # Visualize results.
        E = np.exp(x_final[0])
        nu = np.exp(x_final[1])
        env_opt = SoftStarfishEnv3d(seed, folder, {
            'youngs_modulus': E,
            'poissons_ratio': nu,
            'act_stiffness': act_stiffness,
            'y_actuator': False,
            'z_actuator': True,
            'fix_center_x': False,
            'fix_center_y': True,
            'fix_center_z': True,
            'use_stepwise_loss': True,
            'data': measurement_data,
            'substep': substep,
            'render_markers': True,
            'spp': spp
        })
        env_opt.simulate(dt, frame_num, pd_method, pd_opt, q0, v0, actuator_signals, f0,
            require_grad=False, vis_folder='sys_id_render', velocity_bound=max_vel)
        export_gif(folder / 'sys_id_render', '{}.gif'.format('sys_id_render'), fps=int(1 / dt))

        ###########################################################################
        # Trajectory optimization
        ###########################################################################
        # Create an environment with the final material parameters and with all 4 legs
        env_final = SoftStarfishEnv3d(seed, folder, {
            'youngs_modulus': E,
            'poissons_ratio': nu,
            'act_stiffness': act_stiffness,
            'y_actuator': True,
            'z_actuator': True,
            'fix_center_x': False,
            'fix_center_y': True,
            'fix_center_z': True,
            'use_stepwise_loss': False,
            'data': measurement_data,
            'substep': substep,
            'render_markers': False,
            'spp': spp
        })
        deformable = env_final.deformable()

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = env.default_init_position()
        v0 = np.zeros(dofs)

        # Hold actuator signals constant for finer time step
        control_skip_frame_num = substep
        control_frame_num = int(frame_num // control_skip_frame_num)

        f0 = np.zeros((frame_num, dofs))

        def variable_to_act(x):
            u_full = []
            for i in range(control_frame_num):
                ui_begin = x[i]
                ui_end = x[(i + 1) % control_frame_num]
                for j in range(control_skip_frame_num):
                    t = j / control_skip_frame_num
                    ui = (1 - t) * ui_begin + t * ui_end
                    u = np.zeros(act_dofs)
                    u[:] = ui
                    u_full.append(u)
            return u_full

        x_final = load_latest_data(folder, 'traj_opt')[-1][0]
        # Visualize results.
        a_final = ndarray(variable_to_act(x_final))
        env_final.simulate(dt, frame_num, pd_method, pd_opt, q0, v0, a_final, f0,
            require_grad=False, vis_folder='traj_opt_render', velocity_bound=max_vel)
        export_gif(folder / 'traj_opt_render', '{}.gif'.format('traj_opt_render'), fps=int(1 / dt))
