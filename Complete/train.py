# Import thư viện
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Đọc dữ liệu csv
data = pd.read_csv(r"E:\a-z alphabets\A_Z Handwritten Data.csv").astype('float32')

# Chia dữ liệu thành X - Dữ liệu của chúng tôi và y - nhãn dự đoán
X = data.drop('0', axis=1)
y = data['0']

# Định hình lại dữ liệu
X = np.reshape(X.values, (X.shape[0], 28, 28, 1))

# Chuyển đổi nhãn thành giá trị phân loại
y = to_categorical(y, num_classes=26, dtype='int')

# Tạo một từ điển để ánh xạ ký tự
word_dict = {i: chr(65 + i) for i in range(26)}

# Phân chia thử nghiệm;
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Mô hình CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(26, activation="softmax"))

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# Huấn luyện mô hình
history = model.fit(train_x, train_y, epochs=7, callbacks=[reduce_lr, early_stop], validation_data=(test_x, test_y))

# Đánh giá mô hình cho từng nhân vật và trên toàn bộ tập kiểm tra
individual_accuracies = {}
loss, accuracy = model.evaluate(test_x, test_y)
print(f"Độ mất mát trên tập kiểm tra: {loss:.4f}")
print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")

for i in range(26):
    char_indices = np.where(test_y[:, i] == 1)[0]
    char_X_test = test_x[char_indices]
    char_y_test = test_y[char_indices]

    char_loss, char_accuracy = model.evaluate(char_X_test, char_y_test)
    individual_accuracies[word_dict[i]] = char_accuracy

    print(f'letter {word_dict[i]}: Loss : {char_loss:.4f}, Accurary : {char_accuracy * 100:.2f}%')

# In số liệu đào tạo và xác nhận
print("The validation accuracy is:", history.history['val_accuracy'])
print("The training accuracy is:", history.history['accuracy'])
print("The validation loss is:", history.history['val_loss'])
print("The training loss is:", history.history['loss'])

# Lưu mô hình
model.save(r'model_hand.h5')

# Đánh giá mô hình bằng biểu đồ
# Lấy dữ liệu từ lịch sử huấn luyện
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

# Vẽ biểu đồ loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Train loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Vẽ biểu đồ accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b', label='Train accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
