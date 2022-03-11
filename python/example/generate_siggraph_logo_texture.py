import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
from PIL import Image

from py_diff_pd.common.common import ndarray
from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    folder = Path(root_path)

    # Load images.
    texture_file_name = folder / 'asset/texture/siggraph_logo.jpg'
    img = Image.open(texture_file_name)
    img_data = ndarray(img.getdata()).reshape(img.size[0], img.size[1], 3) / 255
    full_size = 2400
    edge_size = 1
    for s in (25, 50, 75, 100):
        img_resized_data = ndarray(img.resize((s, s)).getdata()).reshape(s, s, 3) / 255
        full_data = ndarray(np.zeros((full_size, full_size, 3)))
        voxel_size = int(full_size // s)
        for i in range(s):
            for j in range(s):
                full_data[i * voxel_size : (i + 1) * voxel_size, j * voxel_size : (j + 1) * voxel_size] \
                    = img_resized_data[i, j]
                full_data[i * voxel_size : i * voxel_size + edge_size, j * voxel_size : (j + 1) * voxel_size] = 0
                full_data[(i + 1) * voxel_size - edge_size : (i + 1) * voxel_size, j * voxel_size : (j + 1) * voxel_size] = 0
                full_data[i * voxel_size : (i + 1) * voxel_size, j * voxel_size : j * voxel_size + edge_size] = 0
                full_data[i * voxel_size : (i + 1) * voxel_size, (j + 1) * voxel_size - edge_size : (j + 1) * voxel_size] = 0

        # Save image.
        Image.fromarray((full_data * 255).astype(dtype=np.uint8), 'RGB').save(
            folder / 'asset/texture/siggraph_logo_{:d}x{:d}.png'.format(s, s))
