import sys
sys.path.append('../')

from pathlib import Path

from py_diff_pd.common.tet_mesh import tetrahedralize, read_tetgen_file, generate_tet_mesh, tet2obj
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import print_info, ndarray
from py_diff_pd.core.py_diff_pd_core import TetMesh3d

import shutil
import os
import numpy as np

if __name__ == '__main__':
    ele_file_name = Path(root_path) / 'asset' / 'mesh' / 'armadillo_10k.ele'
    node_file_name = Path(root_path) / 'asset' / 'mesh' / 'armadillo_10k.node'
    verts, elements = read_tetgen_file(node_file_name, ele_file_name)
    print('Read {:4d} vertices and {:4d} elements'.format(verts.shape[0], elements.shape[0]))
    # To make the mesh consistent with our coordinate system, we need to:
    # - rotate the model along +x by 90 degrees.
    # - shift it so that its min_z = 0.
    # - divide it by 1000.
    R = ndarray([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    verts = verts @ R.T
    # Next, rotate along z by 180 degrees.
    R = ndarray([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ])
    verts = verts @ R.T
    min_z = np.min(verts, axis=0)[2]
    verts[:, 2] -= min_z
    verts /= 1000
    # Now verts and elements are ready to use.

    # Uncomment the follow code if you want to visualize the mesh.
    # tmp_bin_file_name = '.tmp.bin'
    # generate_tet_mesh(verts, elements, tmp_bin_file_name)
    # mesh = TetMesh3d()
    # mesh.Initialize(str(tmp_bin_file_name))
    # tet2obj(mesh, 'armadillo_10k.obj')
    # os.remove(tmp_bin_file_name)

    obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'armadillo_low_res.obj'
    verts, elements = tetrahedralize(obj_file_name, visualize=True, options={
        'minratio': 1.1 })
    print('Generated {:4d} vertices and {:4d} elements'.format(verts.shape[0], elements.shape[0]))