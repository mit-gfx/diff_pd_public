import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
from PIL import Image

from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    folder = Path(root_path)

    color_and_name = [
        ('red', 'E50000'),
        ('green', '15B01A'),
        ('blue', '069AF3'),
        ('orange', 'FFA500'),
        ('black', '000000'),
        ('white', 'F0FFFF')
    ]

    for c, n in color_and_name:
        texture_file_name = folder / 'asset/texture' / '{}.png'.format(c)
        r, g, b = int(n[:2], 16), int(n[2:4], 16), int(n[4:], 16)
        # You can use the code below to create a grid.png.
        edge_width = 4
        img_size = 256
        data = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        data[:, :, 0] = r
        data[:, :, 1] = g
        data[:, :, 2] = b
        data[:edge_width, :, :] = 0
        data[-edge_width:, :, :] = 0
        data[:, :edge_width, :] = 0
        data[:, -edge_width:, :] = 0
        img = Image.fromarray(data, 'RGB')
        img.save(texture_file_name)