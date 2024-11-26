from skimage import transform as transform
from skimage.feature import (match_descriptors, corner_peaks,
                             corner_harris, plot_matches, BRIEF)
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pylab as pylab

# Đọc ảnh
img1 = rgb2gray(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\ronaldo.jpg"))

# Áp dụng phép biến đổi affine và xoay
affine_trans = transform.AffineTransform(scale=(1.2, 1.2), translation=(0, -100))
img2 = transform.warp(img1, affine_trans)
img3 = transform.rotate(img1, 25)

# Phát hiện góc bằng phương pháp Harris
coords1, coords2, coords3 = corner_harris(img1), corner_harris(img2), corner_harris(img3)
coords1[coords1 > 0.01 * coords1.max()] = 1
coords2[coords2 > 0.01 * coords2.max()] = 1
coords3[coords3 > 0.01 * coords3.max()] = 1

# Lọc các góc có thể chấp nhận
keypoints1 = corner_peaks(coords1, min_distance=5)
keypoints2 = corner_peaks(coords2, min_distance=5)
keypoints3 = corner_peaks(coords3, min_distance=5)

# Sử dụng BRIEF để trích xuất đặc trưng
extractor = BRIEF()
extractor.extract(img1, keypoints1)
keypoints1, descriptors1 = keypoints1[extractor.mask], extractor.descriptors
extractor.extract(img2, keypoints2)
keypoints2, descriptors2 = keypoints2[extractor.mask], extractor.descriptors
extractor.extract(img3, keypoints3)
keypoints3, descriptors3 = keypoints3[extractor.mask], extractor.descriptors

# Khớp các mô tả từ ảnh gốc và ảnh đã biến đổi
matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)

# Vẽ kết quả khớp các đặc trưng
fig, axes = pylab.subplots(nrows=2, ncols=1, figsize=(20, 20))
pylab.gray()
plot_matches(axes[0], img1, img2, keypoints1, keypoints2, matches12)
axes[0].axis('off')
axes[0].set_title("Original Image vs. Transformed Image")
plot_matches(axes[1], img1, img3, keypoints1, keypoints3, matches13)
axes[1].axis('off')
axes[1].set_title("Original Image vs. Rotated Image")
pylab.show()