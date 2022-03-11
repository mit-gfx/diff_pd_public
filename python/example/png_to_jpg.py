import sys
import os
from pathlib import Path
from PIL import Image

if __name__ == '__main__':
    img_folder = sys.argv[1]
    for subdir, dirs, files in os.walk(img_folder):
        for filename in files:
            filepath = Path(subdir) / filename
            if str(filepath).endswith('.png'):
                os.system('convert {} {}'.format(str(filepath), str(Path(subdir) / '{}.jpg'.format(filename[:-4]))))
                os.remove(filepath)