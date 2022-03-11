import sys
sys.path.append('../')

from pathlib import Path
import numpy as np

from py_diff_pd.common.common import create_folder, print_info
from py_diff_pd.common.hex_mesh import hex2obj, hex2obj_with_textures
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.env.cantilever_env_3d import CantileverEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('cantilever_3d')
    env = CantileverEnv3d(seed, folder, { 'refinement': 8, 'spp': 64 })
    deformable = env.deformable()

    method = 'pd_eigen'
    opt = { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
            'use_bfgs': 1, 'bfgs_history_size': 10 }

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = np.random.uniform(size=act_dofs)
    f0 = np.random.normal(scale=0.1, size=dofs) * 1e-3

    dt = 1e-2
    frame_num = 25
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