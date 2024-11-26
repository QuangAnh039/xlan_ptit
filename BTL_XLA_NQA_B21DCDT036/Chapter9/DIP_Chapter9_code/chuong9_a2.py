from sklearn import cluster
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize  # Thay thế scipy.misc.imresize đã bị loại bỏ
import numpy as np
import matplotlib.pylab as pylab

# Đọc ảnh và thay đổi kích thước
im = resize(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\sea_bird.jpg"),
            (100, 100, 3), anti_aliasing=True)
img = rgb2gray(im)

# Số cụm (clusters) cho phân đoạn
k = 2  # Phân đoạn nhị phân, với 2 cụm/kết quả

# Chuyển đổi ảnh thành mảng 2D cho mô hình KMeans
X = np.reshape(im, (-1, im.shape[-1]))

# Áp dụng KMeans clustering
two_means = cluster.MiniBatchKMeans(n_clusters=k, random_state=10)
two_means.fit(X)
y_pred = two_means.predict(X)
labels = np.reshape(y_pred, im.shape[:2])

# Hiển thị kết quả KMeans
pylab.figure(figsize=(20, 20))
pylab.subplot(221)
pylab.imshow(np.reshape(y_pred, im.shape[:2]))
pylab.title('k-means segmentation (k=2)', size=30)

pylab.subplot(222)
pylab.imshow(im)
pylab.contour(labels == 0, levels=[0], colors='red')  # Thay 'contours' bằng 'levels'
pylab.axis('off')
pylab.title('k-means contour (k=2)', size=30)

# Áp dụng Spectral Clustering
spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack',
                                      affinity="nearest_neighbors", n_neighbors=10, random_state=10)
spectral.fit(X)
y_pred = spectral.labels_.astype(int)  # Thay np.int bằng int
labels = np.reshape(y_pred, im.shape[:2])

# Hiển thị kết quả Spectral Clustering
pylab.subplot(223)
pylab.imshow(np.reshape(y_pred, im.shape[:2]))
pylab.title('spectral segmentation (k=2)', size=30)

pylab.subplot(224)
pylab.imshow(im)
pylab.contour(labels == 0, levels=[0], colors='red')  # Thay 'contours' bằng 'levels'
pylab.axis('off')
pylab.title('spectral contour (k=2)', size=30)

pylab.tight_layout()
pylab.show()