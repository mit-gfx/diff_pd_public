import sys
sys.path.append('../')

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np

from py_diff_pd.common.common import print_info, print_error

if __name__ == '__main__':
    folder = Path('plant_3d')
    try:
        data = pickle.load(open('plant_3d/data_0008_threads.bin', 'rb'))
    except:
        print_error('Log file not found.')
    loss_l, loss_h = data['loss_range']
    print_info('Loss range: {:3f}, {:3f}'.format(loss_l, loss_h))
    def normalize_loss(unnormalized_loss):
        return (unnormalized_loss - loss_l) / (loss_h - loss_l)

    for thread_ct in [8,]:
        data_file = Path('plant_3d') / 'data_{:04d}_threads.bin'.format(thread_ct)
        if data_file.exists():
            print_info('Loading {}'.format(data_file))
            data = pickle.load(open(data_file, 'rb'))
            for method in ['newton_pcg', 'newton_cholesky', 'pd_eigen']:
                total_time = 0
                avg_forward = 0
                average_backward = 0
                for d in data[method]:
                    d['loss'] = normalize_loss(d['loss'])
                    print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                        d['loss'], np.linalg.norm(d['grad']), d['E'], d['nu'], d['forward_time'], d['backward_time']))
                    total_time += d['forward_time'] + d['backward_time']
                    average_backward += d['backward_time']
                    avg_forward += d['forward_time']
                avg_forward /= len(data[method])
                average_backward /= len(data[method])
                print_info('Optimizing with {} finished in {:6.3f}s with {:d} iterations. Average Backward time: {:6.3f}s, Average Forward Time = {:6.3f}s'.format(
                    method, total_time, len(data[method]), average_backward, avg_forward))

    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=24)             # Controls default text sizes.
    plt.rc('axes', titlesize=28)        # Fontsize of the axes title.
    plt.rc('axes', labelsize=30)        # Fontsize of the x and y labels.
    plt.rc('xtick', labelsize=24)       # Fontsize of the tick labels.
    plt.rc('ytick', labelsize=24)       # Fontsize of the tick labels.
    plt.rc('legend', fontsize=28)       # Legend fontsize.
    plt.rc('figure', titlesize=16)      # Fontsize of the figure title.
    Es = {}
    nus = {}
    losses = {}
    for method in ['newton_pcg', 'newton_cholesky', 'pd_eigen']:
        Es[method] = [d['E'] for d in data[method]]
        nus[method] = [d['nu'] for d in data[method]]
        losses[method] = [d['loss'] for d in data[method]]

    fig = plt.figure(figsize=(18, 7))
    opt_iters = np.max([len(Es['newton_pcg']), len(Es['newton_cholesky']), len(Es['pd_eigen'])])
    ax_E = fig.add_subplot(131)
    ax_E.set_position((0.065, 0.27, 0.25, 0.6))
    ax_E.plot([1e6 for _ in range(opt_iters)], linestyle='--', label='Ground truth', color='tab:orange', linewidth=4)

    ax_nu = fig.add_subplot(132)
    ax_nu.set_position((0.41, 0.27, 0.25, 0.6))
    ax_nu.plot([0.4 for _ in range(opt_iters)], linestyle='--', label='Ground truth', color='tab:orange', linewidth=4)

    ax_loss = fig.add_subplot(133)
    ax_loss.set_position((0.745, 0.27, 0.25, 0.6))
    ax_loss.plot([0 for _ in range(opt_iters)], linestyle='--', label='Ground truth', color ='tab:orange', linewidth=4)

    titles = ['Young\'s modulus estimate', 'Poisson\'s ratio estimate', 'loss']
    for title, ax, y in zip(titles, (ax_E, ax_nu, ax_loss), (Es, nus, losses)):
        if 'modulus' in title:
            ax.set_ylabel("Young\'s modulus (Pa)")
            ax.set_yscale('log')
            ax.grid(True)
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif 'Poisson' in title:
            ax.set_ylabel("Poisson\'s ratio")
            ax.grid(True, which='both')
            ax.yaxis.set_minor_formatter(ScalarFormatter())
        else:
            ax.set_ylabel("loss")
            ax.set_yscale('log')
            ax.grid(True)
        ax.set_xlabel('function evaluations')
        for method, method_ref_name, color in zip(['newton_pcg', 'newton_cholesky', 'pd_eigen'],
            ['PCG', 'Cholesky', 'Ours'], ['tab:blue', 'tab:red', 'tab:green']):
            ax.plot(y[method], color=color, label=method_ref_name, linewidth=4)
        ax.set_title(title, pad=25)
        handles, labels = ax.get_legend_handles_labels()

    # Share legends.
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.17))

    fig.savefig(folder / 'plant.pdf')
    plt.show()