import sys
import os
from matplotlib import pyplot as pylab
import cv2  # Dùng để thay đổi kích thước ảnh
import numpy as np

# Thêm thư mục chứa 'model.py' vào sys.path
path_to_model_dir = os.path.join(os.path.dirname(__file__), 'keras-deeplab-v3-plus-master')
sys.path.append(path_to_model_dir)

# Kiểm tra lại sys.path để xác nhận thư mục đã được thêm đúng cách
print("sys.path:", sys.path)

# Import từ model.py trong thư mục 'keras-deeplab-v3-plus-master'
try:
    from model import Deeplabv3
except ImportError as e:
    print("Lỗi khi import: ", e)
    sys.exit(1)

# Khởi tạo mô hình Deeplabv3+
deeplab_model = Deeplabv3()

# Định nghĩa đường dẫn cho ảnh đầu vào và đầu ra
pathIn = 'input'  # Đường dẫn thư mục chứa ảnh đầu vào
pathOut = 'output'  # Đường dẫn thư mục lưu ảnh phân đoạn

# Đọc ảnh đầu vào từ thư mục 'input'
img = pylab.imread(pathIn + "/cycle.jpg")

# Thay đổi kích thước ảnh sao cho chiều dài hoặc chiều rộng không vượt quá 512px
w, h, _ = img.shape
ratio = 512. / np.max([w, h])
resized = cv2.resize(img, (int(ratio * h), int(ratio * w)))

# Chuẩn hóa ảnh (đưa giá trị vào khoảng [-1, 1])
resized = resized / 127.5 - 1.

# Thêm padding để ảnh có kích thước vuông (512x512)
pad_x = int(512 - resized.shape[0])
resized2 = np.pad(resized, ((0, pad_x), (0, 0), (0, 0)), mode='constant')

# Dự đoán phân đoạn ảnh với mô hình Deeplabv3+
res = deeplab_model.predict(np.expand_dims(resized2, 0))

# Lấy nhãn có xác suất cao nhất
labels = np.argmax(res.squeeze(), -1)

# Hiển thị ảnh phân đoạn với cmap 'inferno'
pylab.figure(figsize=(20, 20))
pylab.imshow(labels[:-pad_x], cmap='inferno')
pylab.axis('off')  # Tắt trục
pylab.colorbar()  # Hiển thị thanh màu
pylab.show()

# Lưu kết quả phân đoạn vào thư mục output
pylab.savefig(pathOut + "\\segmented.jpg", bbox_inches='tight', pad_inches=0)
pylab.close()