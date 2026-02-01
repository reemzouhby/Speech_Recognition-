import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import save_model

# ---------------- PARAMETERS ----------------
spectrogram_dir = "spectrograms"
img_size = (128, 128)  # resize for CNN

# ---------------- LOAD DATA ----------------
X = []
y = []

for label in os.listdir(spectrogram_dir):
    folder_path = os.path.join(spectrogram_dir, label)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        img = load_img(file_path, target_size=img_size, color_mode="grayscale")
        img_array = img_to_array(img) / 255.0  # normalize 0-1
        X.append(img_array)
        y.append(label)

X = np.array(X)
y = np.array(y)

# ---------------- ENCODE LABELS ----------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

# ---------------- BUILD CNN ----------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')  # 5 words
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------- TRAIN ----------------
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
model.save("speech_model1.h5")
# ---------------- EVALUATE ----------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# ---------------- PREDICT ----------------
# Example: predict first image in test set
import matplotlib.pyplot as plt
prediction = model.predict(X_test[0:1])
pred_label = le.inverse_transform([np.argmax(prediction)])
plt.imshow(X_test[0].reshape(img_size), cmap='gray')
plt.title(f"Predicted: {pred_label[0]}")
plt.show()
