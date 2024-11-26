import matplotlib.pylab as pylab
from skimage.morphology import skeletonize
from skimage import img_as_float
from skimage.io import imread

# Hàm hiển thị hình ảnh
def plot_image(image, title):
    pylab.imshow(image, cmap='gray')  # Hiển thị ảnh dạng thang độ xám
    pylab.title(title)
    pylab.axis('off')  # Ẩn trục

def plot_images_horizontally(original, filtered, filter_name, sz=(18, 7)):
    pylab.gray()  # Thiết lập chế độ hiển thị ảnh xám
    pylab.figure(figsize=sz)  # Kích thước cửa sổ hiển thị
    pylab.subplot(1, 2, 1)  # Chia cửa sổ hiển thị thành 1 hàng, 2 cột
    plot_image(original, 'original')  # Hiển thị ảnh gốc
    pylab.subplot(1, 2, 2)  # Vị trí hiển thị ảnh thứ 2
    plot_image(filtered, filter_name)  # Hiển thị ảnh sau xử lý
    pylab.show()

# Đọc ảnh và xử lý nhị phân
im = img_as_float(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\butterfly.png", as_gray=True))
threshold = 0.5
im[im <= threshold] = 0
im[im > threshold] = 1

# Tạo skeleton
skeleton = skeletonize(im)

# Hiển thị ảnh gốc và skeleton
plot_images_horizontally(im, skeleton, 'skeleton', sz=(18, 9))