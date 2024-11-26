from skimage.color import rgb2gray  # Import rgb2gray from skimage.color
from skimage.io import imread
from skimage.filters import laplace
import numpy as np
import matplotlib.pyplot as plt
import pylab

# Định nghĩa hàm plot_image để hiển thị ảnh
def plot_image(img, title):
    plt.imshow(img, cmap='gray')  # Hiển thị ảnh với màu xám
    plt.title(title)  # Đặt tiêu đề cho ảnh
    plt.axis('off')  # Ẩn trục (axis)

# Đọc ảnh và chuyển đổi sang ảnh đen trắng (gray)
im = rgb2gray(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\eagle.jpg"))

# Áp dụng bộ lọc Laplace và làm sắc nét ảnh
im1 = np.clip(laplace(im) + im, 0, 1)

# Hiển thị ảnh gốc và ảnh sau khi làm sắc nét
pylab.figure(figsize=(20, 30))
pylab.subplot(211), plot_image(im, 'original image')
pylab.subplot(212), plot_image(im1, 'sharpened image')
pylab.tight_layout()
pylab.show()