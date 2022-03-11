import sys
sys.path.append('../')

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from py_diff_pd.common.common import PrettyTabular
from py_diff_pd.common.common import print_info, print_warning

def transpose_list(l, row_num, col_num):
    assert len(l) == row_num * col_num
    l2 = []
    for j in range(col_num):
        for i in range(row_num):
            l2.append(l[j + i * col_num])
    return l2

if __name__ == '__main__':
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=30)             # Controls default text sizes.
    plt.rc('axes', titlesize=28)        # Fontsize of the axes title.
    plt.rc('axes', labelsize=30)        # Fontsize of the x and y labels.

    folder = Path('rolling_sphere_3d')
    rel_tols, forward_times, backward_times, losses, grads = pickle.load(open(folder / 'table.bin', 'rb'))
    # Check out if there are any corrupted data.
    max_rel_tol_nums = len(rel_tols)
    for method in forward_times:
        if len(forward_times[method]) < max_rel_tol_nums:
            max_rel_tol_nums = len(forward_times[method])
    for method in backward_times:
        if len(backward_times[method]) < max_rel_tol_nums:
            max_rel_tol_nums = len(backward_times[method])
    for method in losses:
        if len(losses[method]) < max_rel_tol_nums:
            max_rel_tol_nums = len(losses[method])
    for method in grads:
        if len(grads[method]) < max_rel_tol_nums:
            max_rel_tol_nums = len(grads[method])
    if max_rel_tol_nums < len(rel_tols):
        print_warning('Rolling sphere data corrupted. Showing the first {} only.'.format(max_rel_tol_nums))
        rel_tols = rel_tols[:max_rel_tol_nums]

    thread_cts = [2, 4, 8]
    forward_backward_times = {}
    for method in forward_times:
        forward_backward_times[method] = np.zeros(len(rel_tols))

    grad_norms = {}
    for method in grads:
        grad_norms[method] = [np.linalg.norm(x) for x in grads[method]]

    for idx, rel_tol in enumerate(rel_tols):
        print_info('rel_tol: {:3.3e}'.format(rel_tol))
        tabular = PrettyTabular({
            'method': '{:^30s}',
            'forward and backward (s)': '{:3.3f}',
            'forward only (s)': '{:3.3f}',
            'loss': '{:3.3f}',
            '|grad|': '{:3.3f}'
        })
        print_info(tabular.head_string())
        for method in forward_times:
            forward_backward_times[method][idx] = forward_times[method][idx] + backward_times[method][idx]
            print(tabular.row_string({
                'method': method,
                'forward and backward (s)': forward_times[method][idx] + backward_times[method][idx],
                'forward only (s)': forward_times[method][idx],
                'loss': losses[method][idx],
                '|grad|': np.linalg.norm(grads[method][idx]) }))


    fig = plt.figure(figsize=(18, 10))
    ax_fb = fig.add_subplot(131)
    ax_f = fig.add_subplot(132)
    ax_b = fig.add_subplot(133)
    titles = ['forward + backward', 'forward', 'backward']
    ax_poses = [(0.12, 0.42, 0.26, 0.52),
        (0.41, 0.42, 0.26, 0.52),
        (0.70, 0.42, 0.26, 0.52)]
    dash_list =[(2, 5), (5, 2), (5, 0), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2), (5, 5), (5, 2, 1, 2)]
    for ax_pos, title, ax, t in zip(ax_poses, titles, (ax_fb, ax_f, ax_b), (forward_backward_times, forward_times, backward_times)):
        ax.set_position(ax_pos)
        ax.set_xlabel('time (s)')
        ax.set_yscale('log')
        if title == 'forward + backward':
            ax.set_ylabel('convergence threshold')
            ax.set_xticks([0, 100, 200])
        else:
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(rel_tols)
        for method, method_ref_name in zip(['newton_pcg', 'newton_cholesky', 'pd_eigen', 'pd_no_acc'],
            ['PCG', 'Cholesky', 'Ours', 'Ours']):
            if 'eigen' in method:
                color = 'tab:green'
            elif 'pcg' in method:
                color = 'tab:blue'
            elif 'cholesky' in method:
                color = 'tab:red'
            elif 'acc' in method:
                color = 'tab:orange'
            if method == 'pd_no_bfgs' and title != 'backward':
                continue
            for idx, thread_ct in enumerate(thread_cts):
                meth_thread_num = '{}-{}'.format(method_ref_name, thread_ct)
                if 'acc' in method:
                    meth_thread_num = 'Ours-{} (No Acc)'.format(thread_ct)
                ax.plot(t['{}_{}threads'.format(method, thread_ct)], rel_tols, label=meth_thread_num,
                    color=color, dashes=dash_list[idx], linewidth=3)

        ax.grid(True)
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()

    # Share legends.
    row_num = 4
    col_num = len(thread_cts)
    fig.legend(transpose_list(handles, row_num, col_num), transpose_list(labels, row_num, col_num),
        loc='upper center', ncol=col_num, bbox_to_anchor=(0.5, 0.30))
    fig.savefig(folder / 'rolling_sphere.pdf')
    fig.savefig(folder / 'rolling_sphere.png')


    plt.rc('font', size=20)
    plt.rc('axes', titlesize=20)        # Fontsize of the axes title.
    plt.rc('axes', labelsize=20)

    fig_l_g = plt.figure(figsize=(12, 8))
    ax_loss = fig_l_g.add_subplot(121)
    ax_grad = fig_l_g.add_subplot(122)
    ax_loss.set_position((0.09, 0.2, 0.37, 0.5))
    ax_grad.set_position((0.60, 0.2, 0.37, 0.5))
    titles_l_g = ['loss', '|grad|']
    for title, ax, y in zip(titles_l_g, (ax_loss, ax_grad), (losses, grad_norms)):
        ax.set_xlabel('convergence threshold')
        ax.set_ylabel('magnitude')
        ax.set_xscale('log')
        ax.invert_xaxis()
        if 'grad' in title:
            ax.set_yscale('log')
        for method, method_ref_name in zip(['newton_pcg', 'newton_cholesky', 'pd_eigen'], ['PCG', 'Cholesky', 'Ours']):
            if 'pd' in method:
                color = 'tab:green'
            elif 'pcg' in method:
                color = 'tab:blue'
            elif 'cholesky' in method:
                color = 'tab:red'
            meth_thread_num = '{}_{}threads'.format(method, thread_cts[-1])
            ax.plot(rel_tols, y[meth_thread_num], label=method_ref_name, color=color, linewidth=4)
        ax.grid(True, which='major')
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()

    fig_l_g.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.1))
    fig_l_g.savefig(folder / 'rolling_sphere_loss_grad.pdf')
    fig_l_g.savefig(folder / 'rolling_sphere_loss_grad.png')
    plt.show()
