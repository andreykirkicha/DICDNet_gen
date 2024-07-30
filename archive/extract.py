import os
import h5py
import numpy as np
from PIL import Image

with open("test_640geo_dir.txt", "r+") as f:
    # Reading from a file
    num = 0
    for line in f:
        file_path = os.path.join(os.getcwd(), line.strip())  # Get full path
        hf = h5py.File(file_path, 'r')
        for_image = hf['image'][:]
        image = Image.fromarray(np.flip(for_image, axis=0))
        image.save('img_' + str(num) +'.tif')

        num += 1
