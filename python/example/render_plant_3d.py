import sys
sys.path.append('../')

import os
import pickle
from pathlib import Path

import numpy as np

from py_diff_pd.common.common import create_folder, print_info
from py_diff_pd.common.hex_mesh import hex2obj_with_textures
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.env.plant_env_3d import PlantEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('plant_3d')
    youngs_modulus = 1e6
    poissons_ratio = 0.4
    env = PlantEnv3d(seed, folder, {
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'spp': 64 })
    deformable = env.deformable()

    # Optimization parameters.
    method = 'pd_eigen'
    thread_ct = 8
    opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
            'use_bfgs': 1, 'bfgs_history_size': 10 }

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    dt = 1e-2
    frame_num = 3
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    vertex_num = int(dofs // 3)
    f0 = np.zeros((vertex_num, 3))
    f0[:, 0] = -1
    f0 = f0.ravel()
    f0 = [f0 for _ in range(frame_num)]
    _, info = env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False,
        vis_folder='initial_condition')
    # Pick the frame where the center of mass is the highest.    
    q0 = info['q'][-1]
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groudtruth motion.
    frame_num = 200
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]
    env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder='groundtruth')

    # Load results.
    folder = Path('plant_3d')
    thread_ct = 8
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    # Initial guess.
    E_init = data[method][0]['E']
    nu_init = data[method][0]['nu']

    # Final result.
    E_final = data[method][-1]['E']
    nu_final = data[method][-1]['nu']
    print_info('Init: {:6.3e}, {:6.3f}'.format(E_init, nu_init))
    print_info('Final: {:6.3e}, {:6.3f}'.format(E_final, nu_final))

    def simulate(E_opt, nu_opt, vis_folder):
        env_opt = PlantEnv3d(seed, folder, { 'youngs_modulus': E_opt, 'poissons_ratio': nu_opt, 'spp': 64 })
        env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=vis_folder)

    simulate(E_init, nu_init, 'init')
    simulate(E_final, nu_final, 'final')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder):
        create_folder(folder / mesh_folder)
        for i in range(frame_num + 1):
            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            mesh = HexMesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj_with_textures(mesh, obj_file_name=folder / mesh_folder / '{:04d}.obj'.format(i))

    generate_mesh('groundtruth', 'groundtruth_mesh')
    generate_mesh('init', 'init_mesh')
    generate_mesh('final', 'final_mesh')
