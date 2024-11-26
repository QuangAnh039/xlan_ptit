import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio

# Load image using imageio instead of misc
lena = imageio.imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\zebras.jpg")

# Add salt-and-pepper noise to the input image
noise = np.random.random(lena.shape)
lena[noise > 0.9] = 255
lena[noise < 0.1] = 0

# Function to plot the image (assuming plot_image is defined somewhere else)
def plot_image(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')  # Hide axes

plot_image(lena, 'Noisy image')
plt.show()

# Create the figure for displaying filtered images
fig = plt.figure(figsize=(20, 15))
i = 1
for p in range(25, 100, 25):
    for k in range(5, 25, 5):
        plt.subplot(3, 4, i)
        filtered = ndimage.percentile_filter(lena, percentile=p, size=(k, k, 1))
        plot_image(filtered, f'{p} percentile, {k}x{k} kernel')
        i += 1

plt.show()