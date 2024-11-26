import pylab
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# Đọc ảnh và chuyển sang ảnh xám
image = rgb2gray(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\horses.jpg"))

# Áp dụng phương pháp Otsu để tính ngưỡng
thresh = threshold_otsu(image)

# Tạo ảnh nhị phân
binary = image > thresh

# Tạo các subplot để hiển thị
fig, axes = pylab.subplots(nrows=2, ncols=2, figsize=(20, 15))
axes = axes.ravel()

# Đặt các subplot
axes[0], axes[1] = pylab.subplot(2, 2, 1), pylab.subplot(2, 2, 2)
axes[2] = pylab.subplot(2, 2, 3, sharex=axes[0], sharey=axes[0])
axes[3] = pylab.subplot(2, 2, 4, sharex=axes[0], sharey=axes[0])

# Hiển thị ảnh gốc
axes[0].imshow(image, cmap=pylab.cm.gray)
axes[0].set_title('Original', size=20)
axes[0].axis('off')

# Hiển thị histogram
axes[1].hist(image.ravel(), bins=256, density=True)
axes[1].set_title('Histogram', size=20)
axes[1].axvline(thresh, color='r')

# Hiển thị ảnh nhị phân sau khi ngưỡng Otsu
axes[2].imshow(binary, cmap=pylab.cm.gray)
axes[2].set_title('Thresholded (Otsu)', size=20)
axes[2].axis('off')

# Ẩn subplot thứ 4
axes[3].axis('off')

# Hiển thị tất cả các subplot
pylab.tight_layout()
pylab.show()