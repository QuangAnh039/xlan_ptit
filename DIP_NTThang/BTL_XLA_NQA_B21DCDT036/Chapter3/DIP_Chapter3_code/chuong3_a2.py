import numpy as np
from skimage import img_as_float
from scipy import signal
import matplotlib.pyplot as plt  # Thay vì pylab

# Đọc và hiển thị ảnh
im = img_as_float(plt.imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\shutterstock.jpg"))
plt.figure(), plt.imshow(im), plt.axis('off'), plt.show()

# Tạo kernel Gaussian 1D
x = np.linspace(-10, 10, 15)
kernel_1d = np.exp(-0.005 * x ** 2)
kernel_1d /= np.trapz(kernel_1d)  # chuẩn hóa tổng = 1
gauss_kernel1 = kernel_1d[:, np.newaxis] * kernel_1d[np.newaxis, :]

# Tạo kernel Gaussian 2
kernel_1d = np.exp(-5 * x ** 2)
kernel_1d /= np.trapz(kernel_1d)  # chuẩn hóa tổng = 1
gauss_kernel2 = kernel_1d[:, np.newaxis] * kernel_1d[np.newaxis, :]

# Tạo kernel DoG
DoGKernel = gauss_kernel1[:, :, np.newaxis] - gauss_kernel2[:, :, np.newaxis]

# Áp dụng DoG kernel vào ảnh
im = signal.fftconvolve(im, DoGKernel, mode='same')

# Hiển thị ảnh sau khi lọc
plt.figure(), plt.imshow(np.clip(im, 0, 1)), print(np.max(im)), plt.show()