import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# --------------------
# PARAMETERS
# --------------------
IMG_SIZE = 128
CATEGORIES = ["walking", "running", "cycling"]
EPOCHS = 5
BATCH_SIZE = 32

DATA_DIR = "../data/dataset"

# --------------------
# LOAD DATA
# --------------------
def load_images(split):
    X, y = [], []

    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, split, category)

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(img)
            y.append(label)

    X = np.array(X) / 255.0
    y = np.array(y)
    return X, y

print("🔹 Loading dataset...")
X_train, y_train = load_images("train")
X_test, y_test = load_images("test")

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# --------------------
# BUILD CNN MODEL
# --------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------
# TRAIN MODEL
# --------------------
print("🔹 Training started...")
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# --------------------
# SAVE MODEL
# --------------------
model.save("../activity_model.h5")
print("✅ Training completed. Model saved as activity_model.h5")
