import sys
sys.path.append('../')

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec
import numpy as np
import csv

from py_diff_pd.common.common import print_info

if __name__ == '__main__':
    data_folder = Path('seafood_data')
    demos = ['water_snake', 'starfish', 'shark']
    data = {}

    for demo in demos:
        folder = Path(data_folder/'{}'.format(demo))
        data[demo] = {}
        for method in ['ppo', 'diffpd']:
            file_name = Path(folder/'{}.csv'.format(method))
            data[demo][method] = ([], [])
            with open(file_name) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    data[demo][method][0].append(int(row[0]))
                    data[demo][method][1].append(float(row[1]))

    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=30)             # Controls default text sizes.
    # plt.rc('axes', titlesize=28)        # Fontsize of the axes title.
    # plt.rc('axes', labelsize=30)        # Fontsize of the x and y labels.
    # # plt.rc('xtick', labelsize=16)       # Fontsize of the tick labels.
    # # plt.rc('ytick', labelsize=16)       # Fontsize of the tick labels.
    # plt.rc('legend', fontsize=28)       # Legend fontsize.
    # plt.rc('figure', titlesize=16)      # Fontsize of the figure title.

    fig = plt.figure(figsize=(18, 9))

    ax_snake1 = fig.add_subplot(161)
    ax_snake1.set_position((0.06, 0.27, 0.135, 0.68))
    ax_snake2 = fig.add_subplot(162, sharey=ax_snake1)
    ax_snake2.set_position((0.2, 0.27, 0.135, 0.68))

    ax_starfish1 = fig.add_subplot(163, sharey = ax_snake1)
    ax_starfish1.set_position((0.38, 0.27, 0.135, 0.68))
    ax_starfish2 = fig.add_subplot(164, sharey=ax_snake1)
    ax_starfish2.set_position((0.52, 0.27, 0.135, 0.68))

    ax_shark1 = fig.add_subplot(165, sharey=ax_snake1)
    ax_shark1.set_position((0.7, 0.27, 0.135, 0.68))
    ax_shark2 = fig.add_subplot(166, sharey=ax_snake1)
    ax_shark2.set_position((0.84, 0.27, 0.135, 0.68))


    titles = ['Lamprey Demo', 'Starfish Demo', 'Shark Demo']
    for title, axes in zip(titles, ((ax_snake1, ax_snake2), (ax_starfish1, ax_starfish2), (ax_shark1, ax_shark2))):
        ax1 = axes[0]
        ax2 = axes[1]
        if 'Lamprey' in title:
            demo = 'water_snake'
            ax1.set_ylabel("reward")
            ax1.grid(True)
            ax2.grid(True)

        elif 'Starfish' in title:
            demo = 'starfish'
            ax1.grid(True)
            ax2.grid(True)
        else:
            demo = 'shark'
            #ax1.set_ylabel("reward")
            ax1.grid(True)
            ax2.grid(True)

        ax2.set_xlabel('time steps                 ')
        for method, method_ref_name, color in zip(['ppo', 'diffpd'],
            ['DRL', 'DiffPD (Ours)'], ['tab:blue', 'tab:green']):
            x = data[demo][method][0]
            y = data[demo][method][1]
            if 'ppo' in method:
                xlim_list = list(filter(lambda i: i > 2e4, x))
                xlim = xlim_list[1]
                xlim_index = x.index(xlim)

                ax1.plot(x, y, color=color, label=method_ref_name, linewidth=3)
                ax2.plot(x[xlim_index:], y[xlim_index:], color=color, label=method_ref_name, linewidth=2)
            else:
                ax1.plot(x, y, color=color, label=method_ref_name, linewidth=3)

        ax1.set_xlim([0, xlim])
        ax2.set_xlim([xlim, 1e6])

        ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

        ax1.spines['right'].set_linestyle((0, (2,6)))
        ax2.spines['left'].set_linestyle((0, (2,6)))
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()

        d = 0.015 # How large the diagonals at the break are
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1-d,1+d), (-d, d), **kwargs)
        ax1.plot((1-d,1+d), (1-d,1+d), **kwargs)

        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, d), (-d, +d), **kwargs)
        ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

        #ax2.set_title(title, pad=25, loc='right')
        handles, labels = ax1.get_legend_handles_labels()
        ax2.xaxis.labelpad = 35

    plt.setp(ax_snake2.get_yticklabels(), visible=False)
    plt.setp(ax_starfish1.get_yticklabels(), visible=False)
    plt.setp(ax_starfish2.get_yticklabels(), visible=False)
    plt.setp(ax_shark1.get_yticklabels(), visible=False)
    plt.setp(ax_shark2.get_yticklabels(), visible=False)
    #plt.subplots_adjust(left=0.06, right=0.98, bottom=0.3, top=0.95)
    # Share legends.
    fig.legend(handles, labels, loc='lower center', ncol=2)

    fig.savefig(data_folder / 'seafood.pdf')
    fig.savefig(data_folder / 'seafood.png')
    plt.show()
