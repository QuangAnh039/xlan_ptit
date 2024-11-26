from PIL import Image
import matplotlib.pyplot as plt

# Mở ảnh gốc
im = Image.open(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\lena.jpg")

# Tăng kích thước ảnh lên 10 lần sử dụng nội suy bậc ba
im_resized = im.resize((im.width * 10, im.height * 10), Image.BICUBIC)

# Hiển thị ảnh gốc và ảnh phóng to
plt.figure(figsize=(10, 5))

# Ảnh gốc
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(im)
plt.axis("off")

# Ảnh phóng to (Bi-cubic)
plt.subplot(1, 2, 2)
plt.title("Resized Image (Bi-cubic)")
plt.imshow(im_resized)
plt.axis("off")

plt.show()