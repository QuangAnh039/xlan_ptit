import numpy as np
import pylab
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from imageio import imread

pylab.figure(figsize=(15, 10))  # Kích thước figure lớn hơn
pylab.gray()  # Hiển thị ảnh dạng grayscale

# Load and process the image
im = np.mean(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\me8.jpg"), axis=2)

# Create Gaussian kernel
gauss_kernel = np.outer(signal.windows.gaussian(im.shape[0], 5), signal.windows.gaussian(im.shape[1], 5))

# Perform FFT on the image and kernel
freq = fft2(im)
assert freq.shape == gauss_kernel.shape
freq_kernel = fft2(ifftshift(gauss_kernel))

# Convolve in the frequency domain
convolved = freq * freq_kernel
im1 = ifft2(convolved).real

# Plot results
pylab.subplot(2, 3, 1), pylab.imshow(im), pylab.title('Original Image', size=16), pylab.axis('off')
pylab.subplot(2, 3, 2), pylab.imshow(gauss_kernel), pylab.title('Gaussian Kernel', size=16), pylab.axis('off')
pylab.subplot(2, 3, 3), pylab.imshow(im1), pylab.title('Output Image', size=16), pylab.axis('off')
pylab.subplot(2, 3, 4), pylab.imshow((20 * np.log10(0.1 + fftshift(freq))).astype(int))
pylab.title('Original Image Spectrum', size=16), pylab.axis('off')
pylab.subplot(2, 3, 5), pylab.imshow((20 * np.log10(0.1 + fftshift(freq_kernel))).astype(int))
pylab.title('Gaussian Kernel Spectrum', size=16), pylab.axis('off')
pylab.subplot(2, 3, 6), pylab.imshow((20 * np.log10(0.1 + fftshift(convolved))).astype(int))
pylab.title('Output Image Spectrum', size=16), pylab.axis('off')

# Adjust spacing
pylab.subplots_adjust(wspace=0.3, hspace=0.3)  # Tăng khoảng cách giữa các subplot
pylab.show()