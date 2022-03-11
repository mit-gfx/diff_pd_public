import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.hex_mesh import hex2obj_with_textures, filter_hex
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.env.quadruped_env_3d import QuadrupedEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('quadruped_3d')
    refinement = 3
    act_max = 1.0
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    leg_z_length = 2
    body_x_length = 4
    body_y_length = 4
    body_z_length = 1
    env = QuadrupedEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'leg_z_length': leg_z_length,
        'body_x_length': body_x_length,
        'body_y_length': body_y_length,
        'body_z_length': body_z_length,
        'spp': 64 })
    deformable = env.deformable()
    leg_indices = env.leg_indices()
    act_indices = env.act_indices()

    # Optimization parameters.
    thread_ct = 8
    opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    method = 'pd_eigen'

    dt = 1e-2
    # We optimize quadruped for 100 frames but we can use larger frames for verifying the open-loop controller.
    frame_num = 400

    # Load results.
    folder = Path('quadruped_3d')
    thread_ct = 8
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    init_offset = ndarray([0, 0, 0.025])
    q0 = (q0.reshape((-1, 3)) + init_offset).ravel()
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    def variable_to_acts(x):
        A_f, A_b, w = x
        a = [np.zeros(act_dofs) for _ in range(frame_num)]
        for i in range(frame_num):
            for key, indcs in leg_indices.items():
                if key[-1] == 'F':
                    for idx in indcs:
                        if key[0] == 'F':
                            a[i][idx] = act_max * (1 + A_f * np.sin(w * i)) / 2
                        else:
                            a[i][idx] = act_max * (1 + A_b * np.sin(w * i)) / 2
                else:
                    for idx in indcs:
                        if key[0] =='F':
                            a[i][idx] = act_max * (1 - A_f * np.sin(w * i)) / 2
                        else:
                            a[i][idx] = act_max * (1 - A_b * np.sin(w * i)) / 2
        return a

    def simulate(x, vis_folder):
        a = variable_to_acts(x)
        env.simulate(dt, frame_num, method, opt, q0, v0, a, f0, require_grad=False, vis_folder=vis_folder)

    # Initial guess and final results.
    x_init = data[method][0]['x']
    x_final = data[method][-1]['x']

    simulate(x_init, 'init')
    simulate(x_final, 'final')

    # Assemble muscles.
    muscle_idx = act_indices
    not_muscle_idx = []
    all_idx = np.zeros(env.element_num())
    for idx in muscle_idx:
        all_idx[idx] = 1
    for idx in range(env.element_num()):
        if all_idx[idx] == 0:
            not_muscle_idx.append(idx)

    # Reconstruct muscle groups.
    muscle_groups = {}
    for key, val in leg_indices.items():
        muscle_groups[key] = [act_indices[v] for v in val]

    def gather_act(act):
        reduced_act = {}
        for key, val in leg_indices.items():
            reduced_act[key] = act[val[0]]
            for v in val:
                assert act[v] == act[val[0]]
        return reduced_act

    def generate_mesh(vis_folder, mesh_folder, x_var):
        create_folder(folder / mesh_folder)
        act = variable_to_acts(x_var)
        color_num = 11
        duv = 1 / color_num
        for i in range(frame_num):
            frame_folder = folder / mesh_folder / '{:d}'.format(i)
            create_folder(frame_folder)

            # Generate body.bin.
            input_bin_file = folder / vis_folder / '{:04d}.bin'.format(i)
            shutil.copyfile(input_bin_file, frame_folder / 'body.bin')

            # Generate body.obj.
            mesh = HexMesh3d()
            mesh.Initialize(str(frame_folder / 'body.bin'))
            hex2obj_with_textures(mesh, obj_file_name=frame_folder / 'body.obj')

            # Generate action.npy.
            frame_act = gather_act(act[i])
            frame_act_flattened = []
            for _, a in frame_act.items():
                frame_act_flattened.append(a)
            np.save(frame_folder / 'action.npy', ndarray(frame_act_flattened))

            # Generate muscle/
            create_folder(frame_folder / 'muscle')
            cnt = 0
            for key, group in muscle_groups.items():
                sub_mesh = filter_hex(mesh, group)
                a = frame_act[key]
                v = a / act_max
                assert 0 <= v <= 1
                v_lower = np.floor(v / duv)
                if v_lower == color_num: v_lower -= 1
                v_upper = v_lower + 1
                texture_map = ((0, v_lower * duv), (1, v_lower * duv), (1, v_upper * duv), (0, v_upper * duv))
                hex2obj_with_textures(sub_mesh, obj_file_name=frame_folder / 'muscle' / '{:d}.obj'.format(cnt),
                    texture_map=texture_map)
                cnt += 1

            # Generate not_muscle.obj.
            sub_mesh = filter_hex(mesh, not_muscle_idx)
            hex2obj_with_textures(sub_mesh, obj_file_name=frame_folder / 'not_muscle.obj')

    generate_mesh('init', 'init_mesh', x_init)
    generate_mesh('final', 'final_mesh', x_final)