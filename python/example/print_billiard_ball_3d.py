import sys
sys.path.append('../')

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec
import numpy as np

from py_diff_pd.common.common import print_info

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

if __name__ == '__main__':
    folder = Path('billiard_ball_3d')
    try:
        data = pickle.load(open(folder / 'data_0008_threads.bin', 'rb'))
    except:
        print_error('Log file not found.')
    loss_l, loss_h = data['loss_range']
    print_info('Loss range: {:3f}, {:3f}'.format(loss_l, loss_h))
    def normalize_loss(unnormalized_loss):
        return (unnormalized_loss - loss_l) / (loss_h - loss_l)

    for thread_ct in [8,]:
        data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
        if data_file.exists():
            print_info('Loading {}'.format(data_file))
            data = pickle.load(open(data_file, 'rb'))
            for method in ['newton_pcg', 'newton_cholesky', 'pd_eigen']:
                total_time = 0
                avg_forward = 0
                average_backward = 0
                for d in data[method]:
                    d['loss'] = normalize_loss(d['loss'])
                    print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                        d['loss'], np.linalg.norm(d['grad']), d['forward_time'], d['backward_time']))
                    total_time += d['forward_time'] + d['backward_time']
                    average_backward += d['backward_time']
                    avg_forward += d['forward_time']
                avg_forward /= len(data[method])
                average_backward /= len(data[method])
                print_info('Optimizing with {} finished in {:6.3f}s in {:d} iterations. Average Backward time: {:6.3f}s, Average Forward Time = {:6.3f}s'.format(
                    method, total_time,  len(data[method]), average_backward, avg_forward))