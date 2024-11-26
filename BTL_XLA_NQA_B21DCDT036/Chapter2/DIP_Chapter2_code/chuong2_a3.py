import numpy.fft as fp
import numpy as np
import matplotlib.pylab as pylab
from skimage import io
from skimage.color import rgb2gray

# Đọc ảnh
image = io.imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\swans.jpg")

# Nếu ảnh có 4 kênh (RGBA), chỉ lấy 3 kênh đầu tiên (RGB)
if image.shape[2] == 4:
    image = image[:, :, :3]

# Chuyển ảnh sang ảnh xám
im1 = rgb2gray(image)

# Tạo hình vẽ
pylab.figure(figsize=(12, 10))

# Tính toán DFT của ảnh
freq1 = fp.fft2(im1)

# Tính toán ảnh hồi phục từ DFT
im1_ = fp.ifft2(freq1).real

# Hiển thị ảnh gốc
pylab.subplot(2, 2, 1)
pylab.imshow(im1, cmap='gray')
pylab.title('Original Image', size=20)

# Hiển thị phổ FFT (Magnitude)
pylab.subplot(2, 2, 2)
pylab.imshow(20 * np.log10(0.01 + np.abs(fp.fftshift(freq1))), cmap='gray')
pylab.title('FFT Spectrum Magnitude', size=20)

# Hiển thị pha FFT
pylab.subplot(2, 2, 3)
pylab.imshow(np.angle(fp.fftshift(freq1)), cmap='gray')
pylab.title('FFT Phase', size=20)

# Hiển thị ảnh đã hồi phục
pylab.subplot(2, 2, 4)
pylab.imshow(np.clip(im1_, 0, 255), cmap='gray')
pylab.title('Reconstructed Image', size=20)

# Hiển thị tất cả hình
pylab.show()