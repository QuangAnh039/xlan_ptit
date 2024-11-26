from skimage import img_as_float
from skimage.io import imread
from skimage.transform import swirl
import matplotlib.pyplot as plt
from PIL import Image

from PIL.ImageChops import subtract, multiply, screen, difference, add
im1 = Image.open(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\sea.JPG") # tải hai khung hình liên tiếp từ video
im2 = Image.open(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\sea_bird.jpg")
im = difference(im1, im2)  # tính sự khác biệt giữa hai ảnh
im.save(r"C:\Tài liệu học tập\1 Quá trình học tập của tôi (Tài liệu + đánh giá)\Năm 4\Xử lý ảnh\kq1.jpg")  # lưu ảnh kết quả
plt.subplot(311)
plt.imshow(im1)  # hiển thị ảnh khung hình đầu tiên
plt.axis('off')
plt.subplot(312)
plt.imshow(im2)  # hiển thị ảnh khung hình thứ hai
plt.axis('off')
plt.subplot(313)
plt.imshow(im)  # hiển thị ảnh sự khác biệt
plt.axis('off')
plt.show()