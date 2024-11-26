import numpy as np
from skimage import img_as_float
import matplotlib.pyplot as pylab
from skimage.io import imread

# Định nghĩa hàm plot_hist để vẽ histogram cho các kênh RGB
def plot_hist(r, g, b, title):
    pylab.hist(r.ravel(), bins=256, color='red', alpha=0.5, label='Red')
    pylab.hist(g.ravel(), bins=256, color='green', alpha=0.5, label='Green')
    pylab.hist(b.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
    pylab.legend(loc='upper right')
    pylab.title(title)

# Đọc ảnh và chuyển sang kiểu float
im = img_as_float(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\me5.jpg"))

# Áp dụng phép biến đổi gamma (gamma correction)
gamma = 5
im1 = im ** gamma

# Cấu hình hiển thị
pylab.style.use('ggplot')
pylab.figure(figsize=(18, 6))

# Vẽ ảnh gốc và ảnh đã biến đổi gamma
pylab.subplot(141)
pylab.imshow(im)
pylab.axis('off')
pylab.title('Original Image')

pylab.subplot(142)
pylab.imshow(im1)
pylab.axis('off')
pylab.title('Gamma Corrected Image')

# Vẽ histogram cho ảnh gốc (RGB)
pylab.subplot(143)
plot_hist(im[..., 0], im[..., 1], im[..., 2], ' (Input)')
pylab.subplots_adjust(wspace=0.3, hspace=0.3)  # Tăng khoảng cách giữa các subplot
# Vẽ histogram cho ảnh đã biến đổi gamma (RGB)
pylab.subplot(144)
plot_hist(im1[..., 0], im1[..., 1], im1[..., 2], ' (Output)')

# Hiển thị kết quả
pylab.show()
