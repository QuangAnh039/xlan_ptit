import numpy as np
from skimage import img_as_float, color, data, restoration
from skimage.io import imread  # Import imread từ skimage

from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as pylab

# Đọc ảnh và chuyển đổi thành ảnh xám
im = color.rgb2gray(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\horses.jpg"))

# Tạo kernel mờ (PSF)
n = 7
psf = np.ones((n, n)) / n**2
im1 = conv2(im, psf, 'same')

# Thêm nhiễu vào ảnh mờ
im1 += 0.1 * np.random.standard_normal(im.shape)  # astro.std() không được định nghĩa

# Áp dụng bộ lọc Wiener không giám sát
im2, _ = restoration.unsupervised_wiener(im1, psf)

# Hiển thị kết quả
fig, axes = pylab.subplots(nrows=1, ncols=3, figsize=(20, 4), sharex=True, sharey=True)
pylab.gray()
axes[0].imshow(im), axes[0].axis('off'), axes[0].set_title('Original image', size=20)
axes[1].imshow(im1), axes[1].axis('off'), axes[1].set_title('Noisy blurred image', size=20)
axes[2].imshow(im2), axes[2].axis('off'), axes[2].set_title('Self tuned restoration', size=20)
fig.tight_layout()
pylab.show()