from skimage.color import rgb2gray  # Import rgb2gray from skimage.color
from skimage.io import imread
from scipy import ndimage
import pylab
import numpy as np

# Đọc ảnh PNG có 4 kênh (RGBA) và lấy chỉ 3 kênh đầu tiên (RGB)
img_rgba = imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\umbc_lib.jpg")

# Lấy 3 kênh RGB (bỏ kênh alpha)
img_rgb = img_rgba[:, :, :3]

# Chuyển ảnh RGB thành ảnh xám (grayscale)
img = rgb2gray(img_rgb)

# Tạo một hình ảnh với kích thước lớn
fig = pylab.figure(figsize=(25, 15))
pylab.gray()  # Hiển thị kết quả bộ lọc dưới dạng ảnh xám

# Áp dụng bộ lọc Gaussian Laplace với các giá trị sigma khác nhau
for sigma in range(1, 10):
    pylab.subplot(3, 3, sigma)
    img_log = ndimage.gaussian_laplace(img, sigma=sigma)  # Áp dụng Gaussian Laplace
    pylab.imshow(np.clip(img_log, 0, 1))  # Hiển thị ảnh đã lọc, cắt giá trị trong khoảng [0,1]
    pylab.axis('off')  # Tắt trục
    pylab.title('LoG with sigma=' + str(sigma), size=20)  # Đặt tiêu đề cho từng ảnh

# Hiển thị các ảnh
pylab.show()