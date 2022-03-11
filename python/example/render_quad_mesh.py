import sys
sys.path.append('../')

import os
import numpy as np
from pathlib import Path

from py_diff_pd.core.py_diff_pd_core import QuadMesh2d
from py_diff_pd.common.display import display_quad_mesh
from py_diff_pd.common.quad_mesh import generate_rectangle_mesh
from py_diff_pd.common.common import ndarray, print_error

def load_mesh_2d(mesh_file):
    mesh = QuadMesh2d()
    mesh.Initialize(str(mesh_file))
    return mesh

def compare_mesh_2d(mesh1, mesh2):
    # Make sure they have the same size.
    if mesh1.NumOfElements() != mesh2.NumOfElements(): return False
    if mesh1.NumOfVertices() != mesh2.NumOfVertices(): return False
    # Check elements.
    for f in range(mesh1.NumOfElements()):
        fi = ndarray(mesh1.py_element(f))
        fj = ndarray(mesh2.py_element(f))
        if np.max(np.abs(fi - fj)) > 0:
            return False
    # Check vertices.
    vi = ndarray(mesh1.py_vertices())
    vj = ndarray(mesh2.py_vertices())
    if np.max(np.abs(vi - vj)) > 0: return False
    return True

def test_render_quad_mesh(verbose):
    folder = Path('render_quad_mesh')
    cell_nums = (2, 4)
    dx = 0.1
    origin = (0, 0)
    binary_file_name = str(folder / 'rectangle.bin')
    generate_rectangle_mesh(cell_nums, dx, origin, binary_file_name)

    mesh_result = load_mesh_2d(binary_file_name)
    if verbose:
        display_quad_mesh(mesh_result)
    mesh_template = load_mesh_2d(folder / 'rectangle_master.bin')
    if not compare_mesh_2d(mesh_template, mesh_result):
        if verbose:
            print_error('The quad mesh was generated incorrectly.')
        return False

    return True

if __name__ == '__main__':
    verbose = True
    test_render_quad_mesh(verbose)