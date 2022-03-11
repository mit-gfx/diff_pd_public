import sys
sys.path.append('../')

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from py_diff_pd.common.common import ndarray

if __name__ == '__main__':
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=24)             # Controls default text sizes.
    plt.rc('axes', titlesize=20)        # Fontsize of the axes title.
    plt.rc('axes', labelsize=20)        # Fontsize of the x and y labels.
    plt.rc('xtick', labelsize=20)       # Fontsize of the tick labels.
    plt.rc('ytick', labelsize=20)       # Fontsize of the tick labels.
    plt.rc('legend', fontsize=20)       # Legend fontsize.
    plt.rc('figure', titlesize=24)      # Fontsize of the figure title.

    folder = Path('landscape_3d')
    rel_scale = 0.001 / (0.08 * 4)    # This is specific to CantileverEnv3d.
    all_losses, all_grads, ss, all_dqs = pickle.load(open(folder / 'table.bin', 'rb'))

    # Normalize data to obtain relative errors.
    grad_base = {}
    for method in ['semi_implicit', 'pd_eigen']:
        loss_base = np.abs(all_losses[method][0][len(all_losses[method][0]) // 2])
        for loss in all_losses[method]:
            assert np.isclose(loss_base, np.abs(loss[len(loss) // 2]))
        all_losses[method] = ndarray(all_losses[method]) / loss_base

        grad_base[method] = np.abs(all_grads[method][0][len(all_grads[method][0]) // 2][0])
        for grad in all_grads[method]:
            assert np.allclose(grad_base[method], np.abs(grad[len(grad) // 2][0]))

    fig = plt.figure(figsize=(16, 14))
    ax_loss_sample = fig.add_subplot(221)
    for method, method_ref_name, color in zip(['semi_implicit', 'pd_eigen'], ['Semi-Implicit', 'DiffPD (Ours)'], ['tab:cyan', 'tab:green']):
        losses = all_losses[method]
        ax_loss_sample.plot([], [], color=color, label=method_ref_name)
        for loss in losses[:5]:
            ax_loss_sample.plot((ss * rel_scale) * 100, (loss - 1) * 100, color=color, linewidth=2)
    ax_loss_sample.set_xlabel('step size (%)')
    ax_loss_sample.set_ylabel('relative change (%)')
    ax_loss_sample.set_ylim([-200, 200])
    ax_loss_sample.grid(True)

    ax_loss_all = fig.add_subplot(222)
    for method, method_ref_name, color in zip(['semi_implicit', 'pd_eigen'], ['Semi-Implicit', 'DiffPD (Ours)'], ['tab:cyan', 'tab:green']):
        losses = all_losses[method]
        ax_loss_all.plot([], [], color=color, label=method_ref_name)
        loss_mean = np.mean(ndarray(losses), axis=0)
        loss_std = np.std(ndarray(losses), axis=0)
        ax_loss_all.plot((ss * rel_scale) * 100, (loss_mean - 1) * 100, color=color, linewidth=2)
        ax_loss_all.fill_between((ss * rel_scale) * 100, (loss_mean - 1 - loss_std) * 100,
            (loss_mean - 1 + loss_std) * 100, color=color, alpha=0.3, linewidth=0)
    ax_loss_all.set_xlabel('step size (%)')
    ax_loss_all.set_ylim([-200, 200])
    ax_loss_all.grid(True)

    ax_grad_sample = fig.add_subplot(223)
    for method, method_ref_name, color in zip(['semi_implicit', 'pd_eigen'], ['Semi-Implicit', 'DiffPD (Ours)'], ['tab:cyan', 'tab:green']):
        grads = all_grads[method]
        ax_grad_sample.plot([], [], color=color, label=method_ref_name)
        for grad in grads[:5]:
            g = [np.linalg.norm(g[0]) for g in grad]
            ax_grad_sample.plot((ss * rel_scale) * 100, (g / np.linalg.norm(grad_base[method]) - 1) * 100, color=color, linewidth=2)
    ax_grad_sample.set_xlabel('step size (%)')
    ax_grad_sample.set_ylabel('relative change (%)')
    ax_grad_sample.set_ylim([-2.5, 2.5])
    ax_grad_sample.grid(True)

    ax_grad_all = fig.add_subplot(224)
    for method, method_ref_name, color in zip(['semi_implicit', 'pd_eigen'], ['Semi-Implicit', 'DiffPD (Ours)'], ['tab:cyan', 'tab:green']):
        grads = all_grads[method]
        ax_grad_all.plot([], [], color=color, label=method_ref_name)
        g = ndarray([[np.linalg.norm(g[0]) for g in grad] for grad in grads]) / np.linalg.norm(grad_base[method])
        g_mean = np.mean(g, axis=0)
        g_std = np.std(g, axis=0)
        ax_grad_all.plot((ss * rel_scale) * 100, (g_mean - 1) * 100, color=color, linewidth=2)
        ax_grad_all.fill_between((ss * rel_scale) * 100, (g_mean - 1 - g_std) * 100,
            (g_mean - 1 + g_std) * 100, color=color, alpha=0.3, linewidth=0)
    ax_grad_all.set_xlabel('step size (%)')
    ax_grad_all.set_ylim([-2.5, 2.5])
    ax_grad_all.grid(True)
    handles, labels = ax_grad_all.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.06))

    fig.savefig(folder / 'landscape_3d.pdf')
    fig.savefig(folder / 'landscape_3d.png')
    plt.show()