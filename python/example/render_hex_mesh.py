import sys
sys.path.append('../')

import os
from pathlib import Path
import numpy as np
from PIL import Image
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.display import render_hex_mesh
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d

def image_to_numpy_array(img_name):
    img = Image.open(img_name).convert('RGB')
    img_data = ndarray(img.getdata()).reshape(img.size[0], img.size[1], 3) / 255
    return img_data

def compare_images(img_data1, img_data2, abs_tol, rel_tol):
    return all(np.abs(img_data1.ravel() - img_data2.ravel()) <= abs_tol + img_data1.ravel() * rel_tol)

def test_render_hex_mesh(verbose):
    render_ok = True

    folder = Path('render_hex_mesh')
    voxels = np.ones((10, 10, 10))
    bin_file_name = str(folder / 'cube.bin')
    generate_hex_mesh(voxels, 0.1, (0, 0, 0), bin_file_name)
    mesh = HexMesh3d()
    mesh.Initialize(bin_file_name)

    resolution = (400, 400)
    sample_num = 64
    render_hex_mesh(mesh, folder / 'render_hex_mesh_1.png', resolution=resolution, sample=sample_num)
    if verbose:
        os.system('eog {}'.format(folder / 'render_hex_mesh_1.png'))

    # Demonstrate more advanced options.
    resolution = (600, 600)
    sample_num = 16
    # Scale the cube by 0.5, rotate along the vertical axis by 30 degrees, and translate by (0.5, 0.5, 0).
    transforms = [('s', 0.5), ('r', (np.pi / 6, 0, 0, 1)), ('t', (0.5, 0.5, 0))]
    render_hex_mesh(mesh, folder / 'render_hex_mesh_2.png', resolution=resolution, sample=sample_num, transforms=transforms,
        render_voxel_edge=True)
    if verbose:
        os.system('eog {}'.format(folder / 'render_hex_mesh_2.png'))

    return True

if __name__ == '__main__':
    # Use verbose = True by default in all example scripts.
    verbose = True
    test_render_hex_mesh(verbose)