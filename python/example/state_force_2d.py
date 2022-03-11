import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import QuadDeformable, QuadMesh2d, StdRealVector
from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdRealArray2d
from py_diff_pd.core.py_diff_pd_core import GravitationalStateForce2d, QuadHydrodynamicsStateForce
from py_diff_pd.core.py_diff_pd_core import PlanarContactStateForce2d, ArcContactStateForce2d
from py_diff_pd.core.py_diff_pd_core import BilliardBallStateForce2d
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_error, print_ok
from py_diff_pd.common.quad_mesh import generate_rectangle_mesh, get_boundary_edge
from py_diff_pd.common.grad_check import check_gradients

def test_state_force_2d(verbose):
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    if verbose:
        print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e4
    cell_nums = (20, 10)
    dx = 0.2

    # Initialization.
    folder = Path('state_force_2d')
    create_folder(folder)
    bin_file_name = folder / 'rectangle.bin'
    generate_rectangle_mesh(cell_nums, dx, (0, 0), bin_file_name)

    mesh = QuadMesh2d()
    mesh.Initialize(str(bin_file_name))

    deformable = QuadDeformable()
    deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
    deformable.AddStateForce('gravity', [0.0, -9.81])
    deformable.AddStateForce('planar_contact', [0.01, 0.99, -dx / 2, 2, 1e2, 1e2, 0.5])
    deformable.AddStateForce('arc_contact', [2, 2, 0, 0, 1, -1, 2.5, np.pi / 2, 2, 1e2, 0.5, 1e2])
    # Hydrodynamics parameters.
    rho = 1e3
    v_water = [0.1, -0.4]   # Velocity of the water.
    # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
    Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
    # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
    Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
    max_force = 0.1
    # The current Cd and Ct are similar to Figure 2 in SoftCon.
    # surface_faces is a list of (v0, v1) where v0 and v1 are the vertex indices of the two endpoints of a boundary edge.
    # The order of (v0, v1) is determined so that following all v0 -> v1 forms a ccw contour of the deformable body.
    surface_faces = get_boundary_edge(mesh)
    deformable.AddStateForce('hydrodynamics', np.concatenate([[rho,], v_water, Cd_points.ravel(), Ct_points.ravel(), [max_force,],
        ndarray(surface_faces).ravel()]))

    dofs = deformable.dofs()
    q0 = ndarray(mesh.py_vertices()) + np.random.normal(scale=dx * 0.1, size=dofs)
    v0 = np.random.normal(size=dofs)
    f_weight = np.random.normal(size=dofs)

    # Test each state force individually.
    gravity = GravitationalStateForce2d()
    g = StdRealArray2d()
    for i in range(2): g[i] = np.random.normal()
    gravity.PyInitialize(1.2, g)

    planar_collision = PlanarContactStateForce2d()
    normal = StdRealArray2d()
    for i in range(2): normal[i] = np.random.normal()
    planar_collision.PyInitialize(normal, 0.56, 2, 1.2, 3.4, 0.56)

    arc_collision = ArcContactStateForce2d()
    arc_center = StdRealArray2d([2, 2])
    arc_dir = StdRealArray2d([0, 0])
    arc_start = StdRealArray2d([1, -1])
    arc_collision.PyInitialize(arc_center, arc_dir, arc_start, 2.5, np.pi / 2, 2, 1e2, 0.5, 1e2)

    hydro = QuadHydrodynamicsStateForce()
    flattened_surface_faces = [ll for l in surface_faces for ll in l]
    hydro.PyInitialize(rho, v_water, Cd_points.ravel(), Ct_points.ravel(), max_force, flattened_surface_faces)

    if verbose:
        # Visualize Cd and Ct.
        angle = np.linspace(-np.pi / 2, np.pi / 2, 101)
        Cd = []
        Ct = []
        for a in angle:
            Cd.append(hydro.Cd(a))
            Ct.append(hydro.Ct(a))

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.plot(angle, Cd, color='tab:orange', label='Cd')
        ax.plot(angle, Ct, color='tab:green', label='Ct')
        ax.grid(True)
        ax.set_xlabel('Angle of attack')
        ax.set_ylabel('Coeff')
        ax.legend()
        plt.show()

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    for state_force in [gravity, planar_collision, arc_collision, hydro]:
        def l_and_g(x):
            return loss_and_grad(x, f_weight, state_force, dofs)
        if not check_gradients(l_and_g, np.concatenate([q0, v0]), eps, rtol, atol, verbose):
            if verbose:
                print_error('StateForce2d gradients mismatch.')
            return False

    # Test billiard.
    billiard = BilliardBallStateForce2d()
    single_ball_vertex_num = 16
    radius = 0.1
    stiffness = 1e1
    coeff = 0.2
    # Generate q0 and v0.
    ball_num = 3
    billiard.Initialize(radius, single_ball_vertex_num, np.full((ball_num,), stiffness), np.full((ball_num,), coeff))
    billiard_dofs = 2 * single_ball_vertex_num * ball_num
    single_ball_q = [(np.cos(i * np.pi * 2 / single_ball_vertex_num) * radius,
        np.sin(i * np.pi * 2 / single_ball_vertex_num) * radius) for i in range(single_ball_vertex_num)]
    single_ball_q = ndarray(single_ball_q)
    billiard_q0 = np.vstack([
        single_ball_q,
        single_ball_q + ndarray([1.9 * radius, 0]),
        single_ball_q + ndarray([0, 1.9 * radius])
    ]).ravel()
    billiard_v0 = np.random.normal(size=billiard_dofs, scale=0.05)
    billiard_weight = np.random.normal(size=billiard_dofs)
    def l_and_g(qvp):
        q = qvp[:billiard_dofs]
        v = qvp[billiard_dofs:2 * billiard_dofs]
        p = qvp[2 * billiard_dofs:]
        billiard_qvp = BilliardBallStateForce2d()
        billiard_qvp.Initialize(radius, single_ball_vertex_num, p[:ball_num], p[ball_num:])

        f = ndarray(billiard_qvp.PyForwardForce(q, v))
        loss = f.dot(billiard_weight)

        # Compute gradients.
        dl_df = np.copy(billiard_weight)
        dl_dq = StdRealVector(dofs)
        dl_dv = StdRealVector(dofs)
        dl_dp = StdRealVector(billiard_qvp.NumOfParameters())
        billiard_qvp.PyBackwardForce(q, v, f, dl_df, dl_dq, dl_dv, dl_dp)
        grad = np.concatenate([ndarray(dl_dq), ndarray(dl_dv), ndarray(dl_dp)])
        return loss, grad
    if not check_gradients(l_and_g, np.concatenate([billiard_q0, billiard_v0,
        [stiffness, stiffness, stiffness, coeff, coeff, coeff,]]), eps, rtol, atol, verbose):
        if verbose:
            print_error('BilliardBallStateForce gradients mismatch.')
        return False

    # Test it in Deformable.
    def forward_and_backward(qvp):
        q = qvp[:dofs]
        v = qvp[dofs:2 * dofs]
        p = qvp[2 * dofs:]
        p_deformable = QuadDeformable()
        p_deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        p_deformable.AddStateForce('gravity', [p[0], p[1]])
        p_deformable.AddStateForce('planar_contact', [0.01, 0.99, -dx / 2, 2, p[2], p[3], p[4]])
        Cd_points = p[5:13]
        Ct_points = p[13:]
        p_deformable.AddStateForce('hydrodynamics', np.concatenate([[rho,], v_water, Cd_points, Ct_points, [max_force,],
            ndarray(surface_faces).ravel()]))

        f = ndarray(p_deformable.PyForwardStateForce(q, v))
        loss = f.dot(f_weight)
        grad_q = StdRealVector(dofs)
        grad_v = StdRealVector(dofs)
        grad_p = StdRealVector(p_deformable.NumOfStateForceParameters())
        p_deformable.PyBackwardStateForce(q, v, f, f_weight, grad_q, grad_v, grad_p)
        grad = np.zeros(qvp.size)
        grad[:dofs] = ndarray(grad_q)
        grad[dofs:2 * dofs] = ndarray(grad_v)
        grad[2 * dofs:] = ndarray(grad_p)
        return loss, grad

    eps = 1e-8
    atol = 1e-3
    rtol = 1e-2

    p0 = ndarray([0.0, -9.81, 3.4, 1.2, 0.5, 0.0, 0.05, 0.4, 0.05, 0.7, 1.85, 1.0, 2.05, -1, -0.8, -0.3, -0.5, 0.3, 0.1, 1, 2.5])
    x0 = np.concatenate([q0, v0, p0])
    grads_equal = check_gradients(forward_and_backward, x0, eps, rtol, atol, verbose)
    if not grads_equal:
        if verbose:
            print_error('ForwardStateForce and BackwardStateForce do not match.')
        return False

    shutil.rmtree(folder)

    return True

def loss_and_grad(qv, f_weight, state_force, dofs):
    q = qv[:dofs]
    v = qv[dofs:]
    f = ndarray(state_force.PyForwardForce(q, v))
    loss = f.dot(f_weight)

    # Compute gradients.
    dl_df = np.copy(f_weight)
    dl_dq = StdRealVector(dofs)
    dl_dv = StdRealVector(dofs)
    dl_dp = StdRealVector(state_force.NumOfParameters())
    state_force.PyBackwardForce(q, v, f, dl_df, dl_dq, dl_dv, dl_dp)
    grad = np.concatenate([ndarray(dl_dq), ndarray(dl_dv)])
    return loss, grad

if __name__ == '__main__':
    verbose = True
    test_state_force_2d(verbose)
