import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.core.py_diff_pd_core import StdRealVector

if __name__ == '__main__':
    parent_folder = Path('rope_2d')
    methods = ('pd_eigen', 'newton_pcg')

    frame_num = 301
    node_nums = 1025
    full_len = 1024 * 0.005
    radius = 2.5
    center = ndarray([0, 2.5])

    # Color.
    bg_color = ndarray([.95, .95, .95])
    rope_color = ndarray([1., .8, .1])
    obstacle_color = ndarray([.8, .8, .8])
    for ratio in [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]:
        folder = parent_folder / 'ratio_{:3f}'.format(ratio)
        for method in methods:
            method_info = pickle.load(open(folder / '{}.data'.format(method), 'rb'))
            q = method_info['q']
            create_folder(folder / '{}_rendering'.format(method), exist_ok=True)

            # The obstacle patch.
            patches = []
            node_left = int(node_nums * (0.5 - ratio / 2))
            node_right = int(node_nums * (0.5 + ratio / 2)) - 1
            q_last = np.reshape(q[-1], (node_nums, -1, 2))
            q_left = q_last[node_left, 0]
            q_right = q_last[node_right, 0]
            angle_left = np.arctan2((q_left - center)[1], (q_left - center)[0])
            angle_right = np.arctan2((q_right - center)[1], (q_right - center)[0])
            angles = np.linspace(angle_left, angle_right, 50)
            p_vertices = []
            for a in angles:
                c, s = np.cos(a), np.sin(a)
                p_vertices.append(ndarray([radius * c, radius * s]) + center)
            p_vertices.append((q_right[0], 0))
            p_vertices.append((q_left[0], 0))
            p_vertices = ndarray(p_vertices)
            # Draw each frame.
            for i in range(0, frame_num, 100):
                qi = q[i]
                # Render qi.
                img_name = folder / '{}_rendering'.format(method) / '{:04d}.pdf'.format(i)
                fig = plt.figure(figsize=(8, 8))

                # Choose background.
                plt.rcParams['figure.facecolor'] = bg_color
                plt.rcParams['axes.facecolor'] = bg_color
                ax = fig.add_subplot()
                qi = np.reshape(qi, (node_nums, -1, 2))

                # Draw the mesh.
                nx = qi.shape[0]
                ny = qi.shape[1]
                lines = []
                for ii in range(nx - 1):
                    for jj in range(ny):
                        # Draw all horizontal lines.
                        p0 = ndarray(qi[ii, jj])
                        p1 = ndarray(qi[ii + 1, jj])
                        lines.append((p0, p1))
                for ii in range(nx):
                    for jj in range(ny - 1):
                        # Draw all vertical lines.
                        p0 = ndarray(qi[ii, jj])
                        p1 = ndarray(qi[ii, jj + 1])
                        lines.append((p0, p1))
                ax.add_collection(mc.LineCollection(lines, colors=rope_color, linewidth=1.0))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim([-3, 3])
                ax.set_ylim([3, 5.5])
                ax.set_aspect('equal')

                # Draw the obstacle.
                polygon = Polygon(p_vertices.copy(), True)
                patches.append(polygon)
                p = PatchCollection(patches, color=obstacle_color, alpha=.7)
                ax.add_collection(p)

                # Save the figure.
                fig.savefig(img_name, bbox_inches='tight', pad_inches=0)
                plt.close()
            print('{}: forward time: {}s'.format(method, method_info['forward_time']))