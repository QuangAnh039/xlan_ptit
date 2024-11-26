from numpy import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pylab as pylab

# Đọc ảnh và kiểm tra số lượng kênh
im = imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\.png")

# Kiểm tra nếu ảnh có 4 kênh (RGBA) thì chuyển sang RGB
if im.shape[-1] == 4:  # Ảnh có 4 kênh
    im = im[..., :3]  # Loại bỏ kênh alpha

# Chuyển đổi ảnh sang xám
im_gray = rgb2gray(im)

# Áp dụng các thuật toán phát hiện blobs
log_blobs = blob_log(im_gray, max_sigma=30, num_sigma=10, threshold=0.1)
log_blobs[:, 2] = sqrt(2) * log_blobs[:, 2]  # Tính bán kính

dog_blobs = blob_dog(im_gray, max_sigma=30, threshold=0.1)
dog_blobs[:, 2] = sqrt(2) * dog_blobs[:, 2]

doh_blobs = blob_doh(im_gray, max_sigma=30, threshold=0.005)

# Danh sách blobs và thông tin hiển thị
list_blobs = [log_blobs, dog_blobs, doh_blobs]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
sequence = zip(list_blobs, colors, titles)

# Tạo các subplots
fig, axes = pylab.subplots(2, 2, figsize=(20, 20), sharex=True, sharey=True)
axes = axes.ravel()

# Hiển thị ảnh gốc
axes[0].imshow(im, interpolation='nearest')
axes[0].set_title('Original Image', size=30)
axes[0].set_axis_off()

# Hiển thị blobs với các phương pháp khác nhau
for idx, (blobs, color, title) in enumerate(sequence):
    axes[idx + 1].imshow(im, interpolation='nearest')
    axes[idx + 1].set_title('Blobs with ' + title, size=30)
    for blob in blobs:
        y, x, row = blob
        col = pylab.Circle((x, y), row, color=color, linewidth=2, fill=False)
        axes[idx + 1].add_patch(col)
    axes[idx + 1].set_axis_off()

pylab.tight_layout()
pylab.subplots_adjust(wspace=0.1, hspace=0.1)  # Tăng khoảng cách giữa các subplot
pylab.show()