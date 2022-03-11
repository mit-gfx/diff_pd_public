import sys
sys.path.append('../')

from importlib import import_module
from py_diff_pd.common.common import print_ok, print_error

if __name__ == '__main__':
    # If you want to add a new test, simply add its name here --- you can find their names from README.md.
    tests = [
        # Display.
        'render_quad_mesh',
        # Numerical check.
        'state_force_2d',
        'state_force_3d',
        'pd_energy_2d',
        'pd_energy_3d',
        'actuation_2d',
        'actuation_3d',
        'pd_forward',
        'deformable_backward_2d',
        'deformable_backward_3d',
        'collision_2d',
        # Quasi-static solvers.
        'deformable_quasi_static_3d',
    ]

    failure_cnt = 0
    for name in tests:
        test_func_name = 'test_{}'.format(name)
        module_name = name
        test_func = getattr(import_module(module_name), test_func_name)
        if test_func(verbose=False):
            print_ok('[{}] PASSED.'.format(name))
        else:
            print_error('[{}] FAILED.'.format(name))
            failure_cnt += 1
    print('{}/{} tests failed.'.format(failure_cnt, len(tests)))
    if failure_cnt > 0:
        sys.exit(-1)
