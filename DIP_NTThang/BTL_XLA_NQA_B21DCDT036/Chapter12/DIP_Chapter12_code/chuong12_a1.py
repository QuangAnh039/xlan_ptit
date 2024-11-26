import numpy as np
import matplotlib.pyplot as pylab
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import transform

# Đọc ảnh và mặt nạ
image = imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\sea.jpg")
mask_img = rgb2gray(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\sea_bird.jpg"))

# Kiểm tra kích thước ảnh
print(image.shape)

# Hiển thị ảnh gốc và mặt nạ
pylab.figure(figsize=(15, 10))
pylab.subplot(121), pylab.imshow(image), pylab.title('Original Image')
pylab.subplot(122), pylab.imshow(mask_img), pylab.title('Mask for the object to be removed (the dog)')
pylab.show()

# Thực hiện seam carving
pylab.figure(figsize=(10, 12))
pylab.title('Object (the dog) Removed')

# Giả sử bạn đã có một hàm transform.seam_carve để thực hiện seam carving.
out = transform.seam_carve(image, mask_img, 'vertical', 90)
resized = transform.resize(image, out.shape, mode='reflect')

# Hiển thị kết quả
pylab.imshow(out)
pylab.show()