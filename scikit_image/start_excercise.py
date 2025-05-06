import skimage as ski
import matplotlib.pyplot as plt
import matplotlib
from skimage import data
"""
Show available images
"""

image_names = [name for name in dir(ski.data) if not name.startswith('_')]

matplotlib.rcParams['font.size'] = 18
for img in image_names:
    caller = getattr(data,img)
    print(f"Image data loading for {img}")
    if not callable(caller):
        continue
    try:
        image = caller()
        plt.figure()
        plt.title(img)
        if image.ndim == 2:
            plt.imshow(image, cmap=plt.cm.gray)
        else:
            plt.imshow(image)
    except:
        continue

plt.show()
