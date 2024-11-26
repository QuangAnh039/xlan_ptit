from skimage import io  # Nhập thư viện io từ skimage
from skimage.restoration import denoise_tv_chambolle  # Dùng hàm denoise_tv_chambolle
import matplotlib.pyplot as pylab

# Đọc ảnh
image = io.imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\umbc_lib.jpg")

# Hiển thị ảnh gốc và ảnh được làm mượt bằng thuật toán Total Variation Denoising
pylab.figure(figsize=(10, 14))

# Ảnh gốc
pylab.subplot(221)
pylab.imshow(image)
pylab.axis('off')
pylab.title('Original', size=20)

# Làm mượt với trọng số weight = 0.1
denoised_img = denoise_tv_chambolle(image, weight=0.1, channel_axis=-1)
pylab.subplot(222)
pylab.imshow(denoised_img)
pylab.axis('off')
pylab.title('TVD (wt=0.1)', size=20)

# Làm mượt với trọng số weight = 0.2
denoised_img = denoise_tv_chambolle(image, weight=0.2, channel_axis=-1)
pylab.subplot(223)
pylab.imshow(denoised_img)
pylab.axis('off')
pylab.title('TVD (wt=0.2)', size=20)

# Làm mượt với trọng số weight = 0.3
denoised_img = denoise_tv_chambolle(image, weight=0.3, channel_axis=-1)
pylab.subplot(224)
pylab.imshow(denoised_img)
pylab.axis('off')
pylab.title('TVD (wt=0.3)', size=20)

# Hiển thị kết quả
pylab.show()