import cv2
import numpy as np
from keras.models import load_model

# Tạo một từ điển để ánh xạ ký tự
word_dict = {}
for i in range(26):
    word_dict[i] = chr(ord('A') + i)

# Tải mô hình huấn luyện dữ liệu
model = load_model('model_hand.h5')

# Tải hình ảnh cần dự đoán
img = cv2.imread(r'img/C.jpg.png')
img_copy = img.copy()

img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)


img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)


_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)


kernel = np.ones((3, 3), np.uint8)


img_thresh = cv2.dilate(img_thresh, kernel, iterations=1)


# Tìm đường nét của ký tự
contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Chức năng nhận dạng ký tự
def recognize_character(character_image):
    predicted_char = word_dict[np.argmax(model.predict(character_image))]
    return predicted_char


# Duyệt qua từng contour
target_size = (28, 28)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 10:  # Loại bỏ các contour nhỏ
        character_image = img_thresh[y:y + h, x:x + w]

        # Resize ảnh ký tự về kích thước mà mô hình yêu cầu
        character_image = cv2.resize(character_image, target_size)
        character_image = np.reshape(character_image, (1, target_size[0], target_size[1], 1))
        recognized_char = recognize_character(character_image)

        # Vẽ hình chữ nhật xung quanh ký tự
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Đặt ký tự đã nhận diện lên ảnh
        cv2.putText(img, recognized_char, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

# Hiển thị ảnh với các ký tự đã nhận diện
cv2.imshow('Recognized Results', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
