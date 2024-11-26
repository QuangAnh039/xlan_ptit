from skimage import filters
import pylab
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt  # Thêm matplotlib.pyplot

# Định nghĩa hàm plot_image để hiển thị ảnh
def plot_image(img, title):
    plt.imshow(img, cmap='gray')  # Hiển thị ảnh với màu xám
    plt.title(title)  # Đặt tiêu đề cho ảnh
    plt.axis('off')  # Ẩn trục (axis)

# Đọc ảnh và chuyển thành ảnh xám (grayscale)
im = rgb2gray(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\fish.jpg"))  # RGB image to gray scale

# Áp dụng các bộ lọc Sobel
pylab.gray()
pylab.figure(figsize=(20, 18))

pylab.subplot(2, 2, 1)
plot_image(im, 'original')

pylab.subplot(2, 2, 2)
edges_x = filters.sobel_h(im)
plot_image(edges_x, 'sobel_x')

pylab.subplot(2, 2, 3)
edges_y = filters.sobel_v(im)
plot_image(edges_y, 'sobel_y')

pylab.subplot(2, 2, 4)
edges = filters.sobel(im)
plot_image(edges, 'sobel')

# Điều chỉnh khoảng cách giữa các hình ảnh
pylab.subplots_adjust(wspace=0.1, hspace=0.1)

# Hiển thị các hình ảnh
pylab.show()