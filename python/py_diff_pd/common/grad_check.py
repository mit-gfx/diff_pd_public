import numpy as np
from py_diff_pd.common.common import print_error, print_ok

# var_filter
def check_gradients(loss_and_grad, x0, eps=1e-6, rtol=1e-2, atol=1e-4, verbose=True,
    loss_only=None, grad_only=None, skip_var=None):
    if grad_only is None:
        _, grad_analytic = loss_and_grad(x0)
    else:
        grad_analytic = grad_only(x0)

    grads_equal = True
    n = x0.size
    for i in range(n):
        if skip_var is not None and skip_var(i): continue
        x_pos = np.copy(x0)
        x_neg = np.copy(x0)
        x_pos[i] += eps
        x_neg[i] -= eps
        if loss_only is None:
            loss_pos, _ = loss_and_grad(x_pos)
            loss_neg, _ = loss_and_grad(x_neg)
        else:
            loss_pos = loss_only(x_pos)
            loss_neg = loss_only(x_neg)
        grad_numeric = (loss_pos - loss_neg) / 2 / eps
        if not np.isclose(grad_analytic[i], grad_numeric, rtol=rtol, atol=atol):
            grads_equal = False
            if verbose:
                print_error('Variable {}: analytic {}, numeric {}'.format(i, grad_analytic[i], grad_numeric))
            else:
                return grads_equal
        elif verbose:
            print_ok('Variable {} seems good: analytic {}, numeric {}'.format(i, grad_analytic[i], grad_numeric))

    return grads_equal
