import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle
import shutil

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.hex_mesh import filter_hex, hex2obj_with_textures
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.env.routing_tendon_env_3d import RoutingTendonEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('routing_tendon_3d')
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    target = ndarray([0.2, 0.2, 0.45])
    refinement = 2
    muscle_cnt = 4
    muscle_ext = 4
    act_max = 2
    env = RoutingTendonEnv3d(seed, folder, {
        'muscle_cnt': muscle_cnt,
        'muscle_ext': muscle_ext,
        'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'target': target,
        'spp': 64 })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    method = 'pd_eigen'
    opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }

    dt = 1e-2
    frame_num = 100

    # Load results.
    folder = Path('routing_tendon_3d')
    import multiprocessing
    cpu_cnt = multiprocessing.cpu_count()
    thread_ct = cpu_cnt - 1
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]
    act_maps = env.act_maps()
    u_dofs = len(act_maps)
    assert u_dofs * (refinement ** 3) * muscle_ext == act_dofs

    def variable_to_act(x):
        act = np.zeros(act_dofs)
        for i, a in enumerate(act_maps):
            act[a] = x[i]
        return act

    def simulate(x, vis_folder):
        a = variable_to_act(x)
        env.simulate(dt, frame_num, method, opt, q0, v0, [a for _ in range(frame_num)], f0,
            require_grad=False, vis_folder=vis_folder)

    # Initial guess.
    x_init = data[method][0]['x']
    x_final = data[method][-1]['x']
    simulate(x_init, 'init')
    simulate(x_final, 'final')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder, x_val):
        create_folder(folder / mesh_folder)
        color_num = 11
        duv = 1 / color_num
        for i in range(frame_num):
            frame_folder = folder / mesh_folder / '{:d}'.format(i)
            create_folder(frame_folder)

            # Generate body.bin.
            input_bin_file = folder / vis_folder / '{:04}.bin'.format(i)
            shutil.copyfile(input_bin_file, frame_folder / 'body.bin')

            # Generate body.obj.
            mesh = HexMesh3d()
            mesh.Initialize(str(frame_folder / 'body.bin'))
            hex2obj_with_textures(mesh, obj_file_name=frame_folder / 'body.obj')

            # Generate action.npy.
            np.save(frame_folder / 'action.npy', ndarray(x_val))

            # Generate muscle/
            create_folder(frame_folder / 'muscle')
            for j, group in enumerate(act_maps):
                sub_mesh = filter_hex(mesh, group)
                v = x_val[j] / act_max
                assert 0 <= v <= 1
                v_lower = np.floor(v / duv)
                if v_lower == color_num: v_lower -= 1
                v_upper = v_lower + 1
                texture_map = ((0, v_lower * duv), (1, v_lower * duv), (1, v_upper * duv), (0, v_upper * duv))
                hex2obj_with_textures(sub_mesh, obj_file_name=frame_folder / 'muscle' / '{:d}.obj'.format(j),
                    texture_map=texture_map)

    generate_mesh('init', 'init_mesh', x_init)
    generate_mesh('final', 'final_mesh', x_final)

    def save_endpoint_sequences(mesh_folder):
        endpoints = []
        for i in range(frame_num):
            frame_folder = folder / mesh_folder / '{:d}'.format(i)
            mesh = HexMesh3d()
            mesh.Initialize(str(frame_folder / 'body.bin'))
            q = ndarray(mesh.py_vertices())
            endpoint = q.reshape((-1, 3))[-1]
            endpoints.append(endpoint)
        endpoints = ndarray(endpoints)
        np.save(folder / Path(mesh_folder) / '{}_endpoint'.format(mesh_folder), endpoints)

    save_endpoint_sequences('init_mesh')
    save_endpoint_sequences('final_mesh')