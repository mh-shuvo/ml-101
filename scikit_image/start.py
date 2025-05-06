import skimage as ski
import matplotlib.pyplot as plt
# camera = ski.data.camera()

# help(camera.shape)

# plt.imshow(camera)
# plt.axis("off")
# plt.show()

"""
Exploring the 2d and 3d images data how store in the ndarray
retrieving the specific channel data from the spefied image.
Working how show the image using matplotlib using its imshow()
"""
img = ski.data.astronaut()  # RGB image

# Split channels
red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]

# Display each channel
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].imshow(red_channel, cmap='Reds')
axes[0].set_title('Red Channel')
axes[1].imshow(green_channel, cmap='Greens')
axes[1].set_title('Green Channel')
axes[2].imshow(blue_channel, cmap='Blues')
axes[2].set_title('Blue Channel')
axes[3].imshow(img)
axes[3].set_title('Original Channel')

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()