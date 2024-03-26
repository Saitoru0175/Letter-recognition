import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from train import word_dict, test_X, test_yOHE

# Load the trained model
model = load_model('model_hand.h5')

# # Rest of your prediction code here...
# pred = model.predict(test_X[:9])
# print(test_X.shape)
#
#
# # Displaying some of the test images & their predicted labels...
#
# fig, axes = plt.subplots(3,3, figsize=(8,9))
# axes = axes.flatten()
#
# for i,ax in enumerate(axes):
#     img = np.reshape(test_X[i], (28,28))
#     ax.imshow(img, cmap="Greys")
#     pred = word_dict[np.argmax(test_yOHE[i])]
#     ax.set_title("Dự Đoán : "+pred)
#     ax.grid()
#
#
# # Prediction on external image...
#
# img = cv2.imread(r'C:\Users\CrisM\OneDrive\Pictures\Image\was.png')
# img_copy = img.copy()
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (400,440))
#
# img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
# img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
# _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
#
# img_final = cv2.resize(img_thresh, (28,28))
# img_final =np.reshape(img_final, (1,28,28,1))
#
#
# img_pred = word_dict[np.argmax(model.predict(img_final))]
#
# cv2.putText(img, "Dữ Liệu", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = (0,0,230))
# cv2.putText(img, "Dự Đoán : " + img_pred, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,30))
# cv2.imshow('Dataflair handwritten character recognition _ _ _ ', img)
#
#
# while (1):
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()
# Function to recognize a single character
def recognize_character(character_image):
    # Preprocess the character_image as needed
    # ...

    # Predict the character using your model
    predicted_char = word_dict[np.argmax(model.predict(character_image))]
    return predicted_char

# Load the external image
img = cv2.imread(r'C:\Users\CrisM\OneDrive\Pictures\Image\W.png')
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400, 440))

img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours of individual characters
contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

recognized_string = ""

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 10:  # Filter out small contours which may not be characters
        character_image = img_thresh[y:y+h, x:x+w]
        character_image = cv2.resize(character_image, (28, 28))
        character_image = np.reshape(character_image, (1, 28, 28, 1))

        recognized_char = recognize_character(character_image)
        recognized_string += recognized_char

# Display the recognized string
print("Recognized String:", recognized_string)

# Display the image with predictions (you can add this part if needed)
# ...

cv2.putText(img, "Du Lieu ", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = (0,0,230))
cv2.putText(img, "Du Doan : " + recognized_string, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,30))
cv2.imshow('Dataflair handwritten character recognition _ _ _ ', img)
cv2.waitKey(0)
cv2.destroyAllWindows()