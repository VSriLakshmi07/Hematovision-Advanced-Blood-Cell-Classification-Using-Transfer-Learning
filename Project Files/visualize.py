# IMPORT LIBRARIES
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# STEP 1: Load and Prepare the Dataset
data_path = r"C:\Users\dell\Desktop\sri\blood cells\dataset2-master\dataset2-master\images\TEST"
categories = os.listdir(data_path)
IMG_SIZE = 64
X = []
y = []

label_dict = {category: i for i, category in enumerate(categories)}

for category in categories:
    folder = os.path.join(data_path, category)
    label = label_dict[category]
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 2: Preprocess the Data
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# STEP 3: Build and Train the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train for 10 epochs
history = model.fit(X_train, y_train_cat, epochs=10, validation_data=(X_test, y_test_cat))

# STEP 4: Save the Model
model.save("Blood Cell.h5")
print("✅ Model saved successfully as Blood Cell.h5")