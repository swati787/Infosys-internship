import os
import cv2
import numpy as np

data_dir = "../data/dataset"
categories = ["walking", "running", "cycling"]
img_size = 128

def load_images(split):
    X = []
    y = []

    for label, category in enumerate(categories):
        folder = os.path.join(data_dir, split, category)

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

X_train, y_train = load_images("train")
X_test, y_test = load_images("test")

print("✅ Data loaded successfully")
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
