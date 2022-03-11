import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.env.cantilever_env_2d import CantileverEnv2d
from py_diff_pd.core.py_diff_pd_core import StdRealMatrix

def test_pd_energy_2d(verbose):
    seed = 42
    folder = Path('pd_energy_2d')
    refinement = 6
    youngs_modulus = 1e4
    poissons_ratio = 0.45
    env = CantileverEnv2d(seed, folder, { 'refinement': refinement, 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    def loss_and_grad(q):
        loss = deformable.PyComputePdEnergy(q)
        grad = -ndarray(deformable.PyPdEnergyForce(q))
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 5e-2
    q0 = env.default_init_position()
    dofs = deformable.dofs()
    x0 = q0 + np.random.normal(scale=0.01, size=dofs)
    if not check_gradients(loss_and_grad, x0, eps, rtol, atol, verbose):
        if verbose:
            print_error('ComputePdEnergy and PdEnergyForce mismatch.')
        return False

    # Check PdEnergyForceDifferential w.r.t. dq.
    dq = np.random.uniform(low=-1e-6, high=1e-6, size=dofs)
    dw = ndarray([0, 0])
    df_analytical = ndarray(deformable.PyPdEnergyForceDifferential(x0, dq, dw))
    Kq = StdRealMatrix()
    Kw = StdRealMatrix()
    deformable.PyPdEnergyForceDifferential(x0, True, True, Kq, Kw)
    Kq = ndarray(Kq)
    Kw = ndarray(Kw)
    df_analytical2 = Kq @ dq
    if not np.allclose(df_analytical, df_analytical2):
        if verbose:
            print_error('Analytical elastic force differential values do not match.')
        return False

    df_numerical = ndarray(deformable.PyPdEnergyForce(x0 + dq)) - ndarray(deformable.PyPdEnergyForce(x0))
    if not np.allclose(df_analytical, df_numerical, rtol, atol):
        if verbose:
            print_error('Analytical elastic force differential values do not match numerical ones.')
            for a, b in zip(df_analytical, df_numerical):
                if not np.isclose(a, b, rtol, atol):
                    print(a, b, a - b)
        return False

    # Check PdEnergyForceDifferential w.r.t. dw.
    material_params = ndarray([youngs_modulus, poissons_ratio])
    for i in range(2):
        eps = 1e-4
        dmaterial = np.zeros(2)
        dmaterial[i] = eps
        dw = env.material_stiffness_differential(youngs_modulus, poissons_ratio) @ dmaterial
        df_analytical = Kw @ dw
        df_analytical2 = ndarray(deformable.PyPdEnergyForceDifferential(x0, np.zeros(dofs), dw))
        if not np.allclose(df_analytical, df_analytical2): return False

        material_params_pos = np.copy(material_params)
        material_params_pos[i] += eps
        env_pos = CantileverEnv2d(seed, folder, { 'refinement': refinement,
            'youngs_modulus': material_params_pos[0],
            'poissons_ratio': material_params_pos[1] })

        df_numerical = ndarray(env_pos.deformable().PyPdEnergyForce(x0)) \
            - ndarray(deformable.PyPdEnergyForce(x0))
        if not np.allclose(df_analytical, df_numerical, rtol=rtol, atol=atol):
            if verbose:
                print(np.linalg.norm(df_analytical), np.linalg.norm(df_numerical))
                print_error('Analytical and numerical force differential values do not match at w({}).'.format(i))
            return False

    shutil.rmtree(folder)

    return True

if __name__ == '__main__':
    verbose = True
    test_pd_energy_2d(verbose)
