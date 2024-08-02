import numpy as np
from PIL import Image

data = np.flip(np.load('testmask.npy'), axis=0)
for i in range(data.shape[2]):
    image = Image.fromarray(data[..., i].astype(np.float32))

    image.save('mask' + str(i) + '.tif')