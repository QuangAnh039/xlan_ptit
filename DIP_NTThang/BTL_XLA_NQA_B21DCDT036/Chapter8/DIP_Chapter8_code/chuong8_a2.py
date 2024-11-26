import pylab
from skimage.io import imread
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2gray
from skimage.segmentation import mark_boundaries

# Hàm vẽ ảnh
def plot_image(img, title):
    pylab.imshow(img)
    pylab.title(title, size=20)
    pylab.axis('off')

# Đọc ảnh và cắt giảm độ phân giải (downsample)
img = imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\.jpg")[::2, ::2, :3]

# Tạo một figure để vẽ các ảnh
pylab.figure(figsize=(15, 10))

# Vẽ ảnh với các giá trị scale khác nhau
i = 1
for scale in [50, 100, 200, 400]:
    pylab.subplot(2, 2, i)
    segments_fz = felzenszwalb(img, scale=scale, sigma=0.5, min_size=200)
    plot_image(mark_boundaries(img, segments_fz, color=(1, 0, 0)), 'scale=' + str(scale))
    i += 1

# Tiêu đề của toàn bộ figure
pylab.suptitle('Felzenszwalb\'s method', size=30)

# Đảm bảo layout đẹp và không bị chồng lên
pylab.tight_layout(rect=[0, 0.03, 1, 0.95])

# Hiển thị ảnh
pylab.show()