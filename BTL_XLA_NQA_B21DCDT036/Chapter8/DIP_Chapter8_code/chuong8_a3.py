from skimage import segmentation
from skimage import graph  # Thay đổi từ 'skimage.future import graph' thành 'from skimage import graph'
from skimage.io import imread
import numpy as np
import pylab
from skimage.color import label2rgb

# Hàm tính trọng số giữa các vùng (mean color)
def _weight_mean_color(graph, src, dst, n):
    diff = graph._node[dst]['mean color'] - graph._node[src]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

# Hàm hợp nhất màu trung bình
def merge_mean_color(graph, src, dst):
    graph._node[dst]['total color'] += graph._node[src]['total color']
    graph._node[dst]['pixel count'] += graph._node[src]['pixel count']
    graph._node[dst]['mean color'] = (graph._node[dst]['total color'] /
                                      graph._node[dst]['pixel count'])

# Đọc ảnh
img = imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\me5.jpg")

# Phân đoạn ảnh bằng SLIC
labels = segmentation.slic(img, compactness=30, n_segments=400)

# Xây dựng đồ thị RAG với màu sắc trung bình
g = graph.rag_mean_color(img, labels)

# Hợp nhất các vùng dựa trên đồ thị RAG
labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

# Chuyển đổi nhãn thành màu sắc trung bình
out = label2rgb(labels2, img, kind='avg')

# Vẽ các biên giới của các vùng phân đoạn
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

# Hiển thị kết quả
pylab.figure(figsize=(20, 10))
pylab.subplot(121), pylab.imshow(img), pylab.axis('off')
pylab.subplot(122), pylab.imshow(out), pylab.axis('off')
pylab.tight_layout(), pylab.show()