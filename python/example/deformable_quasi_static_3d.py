import sys
sys.path.append('../')

from pathlib import Path
import time
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info, print_error
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.common.display import display_hex_mesh, render_hex_mesh, export_gif

def compare_mesh_3d(mesh1, mesh2):
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
    return np.allclose(vi.ravel(), vj.ravel())

def test_deformable_quasi_static_3d(verbose):
    seed = 42
    np.random.seed(seed)
    if verbose:
        print_info('Seed: {}'.format(seed))

    folder = Path('deformable_quasi_static_3d')
    display_method = 'pbrt'
    render_samples = 16

    # Mesh parameters.
    cell_nums = (2, 2, 4)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.1
    origin = (0, 0, 0)
    bin_file_name = str(folder / 'cube.bin')
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)
    mesh = HexMesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    method = 'newton_cholesky'
    opt = { 'max_newton_iter': 10, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-2, 'verbose': 0, 'thread_ct': 4 }
    deformable = HexDeformable()
    deformable.Initialize(bin_file_name, density, 'corotated', youngs_modulus, poissons_ratio)
    # Boundary conditions.
    theta = np.pi / 6
    c, s = np.cos(theta), np.sin(theta)
    R = ndarray([
        [c, -s],
        [s, c]
    ])
    center = ndarray([cell_nums[0] / 2 * dx, cell_nums[1] / 2 * dx])
    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2]
            vx, vy, vz = mesh.py_vertex(node_idx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
            # Rotate the top nodes.
            node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2] + node_nums[2] - 1
            vx, vy, vz = mesh.py_vertex(node_idx)
            vx_new, vy_new = R @ (ndarray([vx, vy]) - center) + center
            deformable.SetDirichletBoundaryCondition(3 * node_idx, vx_new)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy_new)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)

    # Actuations.
    act_indices = []
    for k in range(cell_nums[2]):
        act_indices.append(k)
    deformable.AddActuation(5e5, [1.0, 0.0, 0.0], act_indices)

    # Quasi-static state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    act = np.zeros(act_dofs)
    f_ext = np.zeros(dofs)
    q_array = StdRealVector(dofs)
    deformable.PyGetQuasiStaticState(method, act, f_ext, opt, q_array)
    deformable.PySaveToMeshFile(q_array, str(folder / 'quasi_static.bin'))
    mesh = HexMesh3d()
    mesh.Initialize(str(folder / 'quasi_static.bin'))
    mesh_template = HexMesh3d()
    mesh_template.Initialize(str(folder / 'quasi_static_master.bin'))
    # Compuare mesh and mesh_template.
    if not compare_mesh_3d(mesh_template, mesh):
        if verbose:
            print_error('Quasi-static solution is not as expected.')
        return False

    if verbose:
        # Display the state.
        if display_method == 'pbrt':
            render_hex_mesh(mesh, file_name=folder / 'quasi_static.png', sample=render_samples,
                transforms=[('t', (.1, .1, 0)), ('s', 2.5)])
            import os
            os.system('eog {}'.format(folder / 'quasi_static.png'))
        elif display_method == 'matplotlib':
            display_hex_mesh(mesh, xlim=[-dx, (cell_nums[0] + 1) * dx], ylim=[-dx, (cell_nums[1] + 1) * dx],
                title='Quasi-static', file_name=folder / 'quasi_static.png', show=True)

    return True

if __name__ == '__main__':
    verbose = True
    test_deformable_quasi_static_3d(verbose)