import sys
sys.path.append('../')

import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from py_diff_pd.common.common import create_folder, print_info, ndarray
from py_diff_pd.common.hex_mesh import hex2obj_with_textures, filter_hex
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.env.torus_env_3d import TorusEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('torus_3d')
    youngs_modulus = 5e5
    poissons_ratio = 0.4
    act_stiffness = 2e5
    act_group_num = 8
    env = TorusEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'act_stiffness': act_stiffness,
        'act_group_num': act_group_num,
        'spp': 64
    })
    deformable = env.deformable()

    # Optimization parameters.
    method = 'pd_eigen'
    thread_ct = 8
    opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct,
            'use_bfgs': 1, 'bfgs_history_size': 10 }

    dt = 4e-3
    frame_num = 400
    control_frame_num = 20
    assert frame_num % control_frame_num == 0

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    init_offset = ndarray([0, 0, 0])
    q0 = (q0.reshape((-1, 3)) + init_offset).ravel()
    v0 = env.default_init_velocity()
    v0 = (v0.reshape((-1, 3)) + ndarray([0.25, 0.0, 0.0])).ravel()
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Compute actuation.
    control_frame = int(frame_num // control_frame_num)
    x_lb = np.zeros(act_group_num * control_frame)
    x_ub = np.ones(act_group_num * control_frame) * 2

    act_groups = env.act_groups()
    def variable_to_act(x):
        x = ndarray(x.ravel()).reshape((control_frame, act_group_num))
        # Linear interpolation.
        x_aug = []
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                x_aug.append((1 - t) * x[c] + t * x[c_next])

        acts = []
        for x_aug_frame in x_aug:
            frame_act = np.zeros(act_dofs)
            for i, group in enumerate(act_groups):
                for j in group:
                    frame_act[j] = x_aug_frame[i]
            acts.append(frame_act)
        acts = ndarray(acts)
        return acts

    def variable_to_gradient(x, dl_dact):
        x = ndarray(x.ravel()).reshape((control_frame, act_group_num))
        # Linear interpolation.
        x_aug = []
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                x_aug.append((1 - t) * x[c] + t * x[c_next])

        grad_x_aug = np.zeros((frame_num, act_group_num))
        for k in range(frame_num):
            x_aug_frame = x_aug[k]
            grad_act = dl_dact[k]
            for i, group in enumerate(act_groups):
                for j in group:
                    grad_x_aug[k, i] += grad_act[j]

        # Backpropagate from grad_x_aug to grad.
        grad = np.zeros(x.shape)
        for c in range(control_frame):
            c_next = c if c == control_frame - 1 else c + 1
            for i in range(control_frame_num):
                t = i * 1.0 / control_frame_num
                grad[c] += (1 - t) * grad_x_aug[c * control_frame_num + i]
                grad[c_next] += t * grad_x_aug[c * control_frame_num + i]

        return grad.ravel()

    # Load results.
    folder = Path('torus_3d')
    thread_ct = 8
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    # Results to be rendered.
    x_random = np.random.uniform(x_lb, x_ub)
    x_init = data[method][0]['x']
    x_final = data[method][-1]['x']

    def simulate(x, vis_folder):
        act = variable_to_act(x)
        env.simulate(dt, frame_num, method, opt, q0, v0, act, f0, require_grad=False, vis_folder=vis_folder)

    def gather_act(act):
        reduced_act = np.zeros(act_group_num)
        for i, group in enumerate(act_groups):
            reduced_act[i] = act[group[0]]
        return reduced_act

    simulate(x_random, 'random')
    simulate(x_init, 'init')
    simulate(x_final, 'final')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder, x_val):
        acts = variable_to_act(x_val)
        acts_lb = variable_to_act(x_lb)
        acts_ub = variable_to_act(x_ub)
        create_folder(folder / mesh_folder)
        for i, act in enumerate(acts):
            frame_folder = folder / mesh_folder / '{:d}'.format(i)
            create_folder(frame_folder)

            # action.npy.
            act_lb = gather_act(acts_lb[i])
            act_ub = gather_act(acts_ub[i])
            act = gather_act(act)
            np.save(frame_folder / 'action.npy', act)

            # body.bin.
            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            shutil.copyfile(mesh_file, frame_folder / 'body.bin')
            # body.obj.
            mesh = HexMesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj_with_textures(mesh, obj_file_name=frame_folder / 'body.obj')

            # muscle/
            create_folder(frame_folder / 'muscle')
            assert len(act_lb) == act_group_num and len(act_ub) == act_group_num and len(act) == act_group_num            
            color_num = 11
            duv = 1 / color_num
            for j, group in enumerate(act_groups):
                v = (act[j] - act_lb[j]) / (act_ub[j] - act_lb[j])
                assert 0 <= v <= 1
                v_lower = np.floor(v / duv)
                if v_lower == color_num: v_lower -= 1
                v_upper = v_lower + 1
                texture_map = ((0, v_lower * duv), (1, v_lower * duv), (1, v_upper * duv), (0, v_upper * duv))
                sub_mesh = filter_hex(mesh, group)
                hex2obj_with_textures(sub_mesh, obj_file_name=frame_folder / 'muscle/{:d}.obj'.format(j), texture_map=texture_map)

    generate_mesh('random', 'random_mesh', x_random)
    generate_mesh('init', 'init_mesh', x_init)
    generate_mesh('final', 'final_mesh', x_final)