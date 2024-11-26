from skimage import img_as_float
from skimage.io import imread
from skimage.util import random_noise
import matplotlib.pyplot as plt

# Đường dẫn sửa lại với raw string
im = img_as_float(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\cheetah.png"))
plt.figure(figsize=(15, 12))
sigmas = [0.1, 0.25, 0.5, 1]

for i in range(4):
    noisy = random_noise(im, var=sigmas[i]**2)
    plt.subplot(2, 2, i + 1)
    plt.imshow(noisy)
    plt.axis('off')
    plt.title('Gaussian noise with sigma=' + str(sigmas[i]), size=20)

plt.tight_layout()
plt.show()