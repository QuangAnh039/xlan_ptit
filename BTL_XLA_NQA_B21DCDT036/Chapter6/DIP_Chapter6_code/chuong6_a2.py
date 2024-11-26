from skimage.morphology import binary_dilation, disk
from skimage import img_as_float
import matplotlib.pylab as pylab
from skimage.io import imread

# Đọc ảnh
im = img_as_float(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\messi.jpg"))

# Kiểm tra số kênh của ảnh
if im.shape[2] == 4:  # Nếu ảnh có 4 kênh (RGBA)
    im = 1 - im[..., 3]  # Lấy kênh alpha
else:  # Nếu ảnh chỉ có 3 kênh (RGB)
    im = im.mean(axis=2)  # Chuyển sang grayscale từ RGB

# Chuyển ảnh về dạng nhị phân
im[im <= 0.5] = 0
im[im > 0.5] = 1

# Hiển thị các kết quả
pylab.gray()
pylab.figure(figsize=(18, 9))

# Ảnh gốc
pylab.subplot(131)
pylab.imshow(im)
pylab.title('Original', size=20)
pylab.axis('off')

# Erosion với dilation với các kích thước khác nhau
for d in range(1, 3):
    pylab.subplot(1, 3, d + 1)
    im1 = binary_dilation(im, disk(2 * d))  # Tạo dilation với hình đĩa (disk) có bán kính 2*d
    pylab.imshow(im1)
    pylab.title('Dilation with disk size ' + str(2 * d), size=20)
    pylab.axis('off')

# Hiển thị kết quả
pylab.show()