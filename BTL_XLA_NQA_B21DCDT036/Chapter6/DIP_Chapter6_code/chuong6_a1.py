from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
import matplotlib.pylab as pylab
from skimage.morphology import binary_erosion, rectangle

# Hàm hiển thị ảnh
def plot_image(image, title=''):
    pylab.title(title, size=20)
    pylab.imshow(image, cmap='gray')
    pylab.axis('off')  # Tắt trục (nếu cần bật, xóa dòng này)

# Đọc ảnh
image = imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\tagore.png")

# Chuyển từ RGBA sang RGB nếu có 4 kênh
if image.shape[-1] == 4:  # Kiểm tra nếu ảnh có kênh alpha
    image = rgba2rgb(image)

# Chuyển sang ảnh xám (grayscale)
im = rgb2gray(image)

# Chuyển thành ảnh nhị phân với ngưỡng cố định
im[im <= 0.5] = 0  # Gán giá trị 0 cho pixel <= 0.5
im[im > 0.5] = 1   # Gán giá trị 1 cho pixel > 0.5

# Hiển thị các kết quả
pylab.gray()
pylab.figure(figsize=(20, 10))

# Ảnh gốc
pylab.subplot(1, 3, 1)
plot_image(im, 'Original')

# Erosion với hình chữ nhật (1, 5)
im1 = binary_erosion(im, rectangle(1, 5))
pylab.subplot(1, 3, 2)
plot_image(im1, 'Erosion with rectangle size (1,5)')

# Erosion với hình chữ nhật (1, 15)
im1 = binary_erosion(im, rectangle(1, 15))
pylab.subplot(1, 3, 3)
plot_image(im1, 'Erosion with rectangle size (1,15)')
pylab.subplots_adjust(wspace=0.7, hspace=0.3)  # Tăng khoảng cách giữa các subplot
# Hiển thị kết quả
pylab.show()