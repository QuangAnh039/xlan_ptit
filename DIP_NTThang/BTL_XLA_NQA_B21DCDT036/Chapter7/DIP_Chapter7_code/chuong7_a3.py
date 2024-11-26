import cv2

# Cung cấp đường dẫn đúng tới thư viện Haar Cascade (chú ý đường dẫn)
opencv_haar_path = './'  # Cung cấp đường dẫn tới thư viện Haar Cascade của opencv
# opencv_haar_path = 'C:/opencv/data/haarcascades/' # Đường dẫn thư viện Haar của opencv cài đặt

# Tải các bộ phân loại Haar cho mặt và mắt
face_cascade = cv2.CascadeClassifier(opencv_haar_path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(opencv_haar_path + 'haarcascade_eye.xml')
# eye_cascade = cv2.CascadeClassifier(opencv_haar_path + 'haarcascade_eye_tree_eyeglasses.xml') # dùng với mắt có kính

# Đọc ảnh
img = cv2.imread(  r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\ronaldo.jpg"    )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Phát hiện mặt
faces = face_cascade.detectMultiScale(gray, 1.2, 5)  # scaleFactor=1.2, minNeighbors=5
print(len(faces))  # In ra số lượng mặt được phát hiện

# Duyệt qua các khuôn mặt và vẽ hình chữ nhật xung quanh
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Phát hiện mắt trong vùng mặt
    eyes = eye_cascade.detectMultiScale(roi_gray)
    print(eyes)  # In ra vị trí của các mắt được phát hiện

    # Vẽ hình chữ nhật quanh mắt
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Lưu ảnh đã phát hiện mặt và mắt
cv2.imwrite('../images/lena_face_detected.jpg', img)