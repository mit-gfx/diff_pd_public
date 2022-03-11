import sys
sys.path.append('../')

from pathlib import Path
import numpy as np

from py_diff_pd.common.common import print_info, ndarray
from py_diff_pd.common.hex_mesh import voxelize, generate_hex_mesh, hex2obj
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    root_folder = Path(root_path)
    torus_bin_file = root_folder / 'asset/mesh/torus_analytic.bin'
    torus_obj_file = root_folder / 'asset/mesh/torus_analytic.obj'

    # Hyperparameters.
    inner_radius = 0.35  # Should be strictly smaller than 0.5.
    width = 0.2 # Needs to be between 0 and 0.5 + inner_radius.
    x_cell_num = 16 # How many voxels would you like the x axis to have.
    # End of hyperparameters.

    outer_radius = 0.5  # Fixed. Do not change.
    assert inner_radius < outer_radius
    assert 0 < width < 1 - (outer_radius - inner_radius)

    x_ext = 2 * outer_radius
    y_ext = outer_radius - inner_radius + width
    z_ext = 2 * outer_radius

    dx = x_ext / x_cell_num
    x_cell_num = int(x_ext / dx) + 1
    y_cell_num = int(y_ext / dx) + 1
    z_cell_num = int(z_ext / dx) + 1

    voxels = np.zeros((x_cell_num, y_cell_num, z_cell_num))
    for i in range(x_cell_num):
        for j in range(y_cell_num):
            for k in range(z_cell_num):
                px, py, pz = ndarray([i + 0.5, j + 0.5, k + 0.5]) * dx
                # Determine whether p is inside or outside the torus.
                inside = False
                # Shift py.
                px -= outer_radius
                pz -= outer_radius
                py -= (outer_radius - inner_radius + width) / 2
                # Rotate.
                theta = np.arctan2(pz, px)
                c, s = np.cos(theta), np.sin(theta)
                R = ndarray([[c, 0, s],
                    [ 0, 1, 0],
                    [ -s, 0, c]])
                px, py, pz = R @ ndarray([px, py, pz])
                assert np.isclose(pz, 0)
                # Symmetry.
                if px < 0: px = -px
                if py < 0: py = -py
                # Case 1:
                if inner_radius <= px <= outer_radius and py <= width / 2:
                    inside = True
                # Case 2:
                elif (px - (inner_radius + outer_radius) / 2) ** 2 + (py - width / 2) ** 2 <= (outer_radius - inner_radius) ** 2 / 4:
                    inside = True
                if inside:
                    voxels[i][j][k] = 1

    # Export data.
    generate_hex_mesh(voxels, dx, [0, -(outer_radius - inner_radius + width) / 2, 0], torus_bin_file, True)
    mesh = HexMesh3d()
    mesh.Initialize(str(torus_bin_file))
    hex2obj(mesh, torus_obj_file, obj_type='tri')
    print_info('Torus elements:', mesh.NumOfElements(), ', DoFs:', 3 * mesh.NumOfVertices())
