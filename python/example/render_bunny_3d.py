import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle

from py_diff_pd.common.common import ndarray, create_folder, rpy_to_rotation, rpy_to_rotation_gradient
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.hex_mesh import hex2obj
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.env.bunny_env_3d import BunnyEnv3d

def apply_transform(q, R, t):
    q = ndarray(q).reshape((-1, 3))
    com = np.mean(q, axis=0)
    return ((q - com) @ R.T + t).ravel()

if __name__ == '__main__':
    seed = 42
    folder = Path('bunny_3d')
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    target_com = ndarray([0.15, 0.15, 0.15])
    env = BunnyEnv3d(seed, folder, {
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'target_com': target_com,
        'spp': 64 })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-3
    frame_num = 100

    # Load results.
    folder = Path('bunny_3d')
    thread_ct = 8
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    def variable_to_initial_states(x):
        init_rpy = x[:3]
        init_R = rpy_to_rotation(init_rpy)
        init_com_q = x[3:6]
        init_com_v = x[6:9]
        init_q = apply_transform(q0, init_R, init_com_q)
        init_v = (v0.reshape((-1, 3)) + init_com_v).ravel()
        return np.copy(init_q), np.copy(init_v)
    def variable_to_initial_states_gradient(x, grad_init_q, grad_init_v):
        grad = np.zeros(x.size)
        # init_rpy:
        offset = q0.reshape((-1, 3)) - np.mean(q0.reshape((-1, 3)), axis=0)
        # init_q = (offset @ R.T + init_com_q).ravel()
        rpy = x[:3]
        dR_dr, dR_dp, dR_dy = rpy_to_rotation_gradient(rpy)
        grad[0] = (offset @ dR_dr.T).ravel().dot(grad_init_q)
        grad[1] = (offset @ dR_dp.T).ravel().dot(grad_init_q)
        grad[2] = (offset @ dR_dy.T).ravel().dot(grad_init_q)
        # init_com_q:
        grad[3:6] = np.sum(grad_init_q.reshape((-1, 3)), axis=0)
        # init_com_v:
        grad[6:9] = np.sum(grad_init_v.reshape((-1, 3)), axis=0)
        return grad

    def simulate(x, method, opt, vis_folder):
        init_q, init_v = variable_to_initial_states(x)
        env.simulate(dt, frame_num, method, opt, init_q, init_v, a0, f0, require_grad=False, vis_folder=vis_folder)

    # Initial guess.
    x = data['newton_pcg'][0]['x']
    simulate(x, methods[0], opts[0], 'init')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder):
        create_folder(folder / mesh_folder)
        for i in range(frame_num + 1):
            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            mesh = HexMesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj(mesh, obj_file_name=folder / mesh_folder / '{:04d}.obj'.format(i), obj_type='tri')

    generate_mesh('init', 'init_mesh')

    for method, opt in zip(methods, opts):
        # Final result.
        x_final = data[method][-1]['x']

        simulate(x_final, method, opt, 'final_{}'.format(method))
        generate_mesh('final_{}'.format(method), 'final_mesh_{}'.format(method))

    def save_com_sequences(mesh_folder):
        coms = []
        for i in range(frame_num + 1):
            mesh_file = folder / mesh_folder / '{:04d}.bin'.format(i)
            mesh = HexMesh3d()
            mesh.Initialize(str(mesh_file))
            q = ndarray(mesh.py_vertices())
            com = np.mean(q.reshape((-1, 3)), axis=0)
            coms.append(com)
        coms = ndarray(coms)
        np.save(folder / '{}_com'.format(mesh_folder), coms)

    save_com_sequences('init')
    for method in methods:
        save_com_sequences('final_{}'.format(method))