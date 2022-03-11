import sys
sys.path.append('../')

from pathlib import Path
import shutil
import os
import numpy as np

from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.hex_mesh import voxelize, generate_hex_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.common.common import print_info, create_folder
from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    folder = Path('pbrt_renderer_demo')
    create_folder(folder)

    # Create a mesh.
    voxels = np.ones((4, 4, 4))
    dx = 0.25
    origin = [0, 0, 0]
    bin_file_name = '.tmp.bin'
    generate_hex_mesh(voxels, dx, origin, bin_file_name)
    mesh = HexMesh3d()
    mesh.Initialize(bin_file_name)
    os.remove(bin_file_name)

    # Render.
    options = {
        'file_name': str(folder / 'demo.png'),
        'light_map': 'uffizi-large.exr',
        'sample': 64,
        'max_depth': 4,
        'camera_pos': (0, -2, 0.8),
        'camera_lookat': (0, 0, 0),
    }
    renderer = PbrtRenderer(options)
    renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=(.3, .7, .5), transforms=[
        ('t', (-1, 0, 0)),
        ('s', 0.25),
        ('r', (1, 0, 1, 0)),
        ('r', (1, 0, 0, 1)),
        ('t', (0, 0, 0.05))
    ])
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
        texture_img='chkbd_24_0.7')
    renderer.render()

    # Display.
    os.system('eog {}'.format(folder / 'demo.png'))