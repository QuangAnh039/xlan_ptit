from PIL import Image
import matplotlib.pylab as pylab
import numpy as np

# Mở ảnh đầu vào
im = Image.open(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\pepper.jpg")  # Đường dẫn ảnh

# Tạo danh sách số lượng màu (giảm dần từ 256 đến 2)
num_colors_list = [1 << n for n in range(8, 0, -1)]  # [256, 128, 64, 32, 16, 8, 4, 2]
snr_list = []  # Danh sách lưu SNR
i = 1

# Thiết lập kích thước hiển thị
pylab.figure(figsize=(20, 30))

# Duyệt qua các số lượng màu
for num_colors in num_colors_list:
    # Lượng tử hóa ảnh
    im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)

    # Hiển thị ảnh đã lượng tử hóa
    pylab.subplot(4, 2, i)  # Hiển thị trên lưới 4x2
    pylab.imshow(im1)  # Hiển thị ảnh
    pylab.axis('off')  # Tắt trục tọa độ

    # Tính SNR (thay thế signaltonoise bằng công thức)
    im_array = np.array(im1)  # Chuyển ảnh thành mảng numpy
    snr = im_array.mean() / im_array.std()  # Tính SNR
    snr_list.append(snr)

    # Đặt tiêu đề
    pylab.title(
        f"Image with # colors = {num_colors}\nSNR = {np.round(snr, 3)}",
        size=20
    )
    i += 1

# Điều chỉnh khoảng cách giữa các ô
pylab.subplots_adjust(wspace=0.2, hspace=0)

# Hiển thị toàn bộ ảnh
pylab.show()