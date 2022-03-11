import sys
sys.path.append('../')

from pathlib import Path
import numpy as np

from py_diff_pd.common.common import create_folder, print_info, ndarray
from py_diff_pd.common.hex_mesh import hex2obj, hex2obj_with_textures
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.env.rolling_sphere_env_3d import RollingSphereEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('rolling_sphere_3d')
    refinement = 10
    youngs_modulus = 2e6
    poissons_ratio = 0.4
    env = RollingSphereEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'spp': 64 })
    deformable = env.deformable()

    method = 'pd_eigen'
    opt = { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': 4,
            'use_bfgs': 1, 'bfgs_history_size': 10 }

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

    env.simulate(dt, frame_num, method, opt, q0, v0, [a0 for _ in range(frame_num)],
        [f0 for _ in range(frame_num)], require_grad=False, vis_folder='groundtruth')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder):
        create_folder(folder / mesh_folder)
        for i in range(frame_num + 1):
            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            mesh = HexMesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj_with_textures(mesh, obj_file_name=folder / mesh_folder / '{:04d}.obj'.format(i))

    generate_mesh('groundtruth', 'groundtruth_mesh')