import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("../activity_model.h5")

# Categories (same order as training)
categories = ["walking", "running", "cycling"]

# Load a test image (change name if needed)
img_path = "../data/frames/walking/frame_0.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_class = categories[np.argmax(prediction)]

print("✅ Predicted activity:", predicted_class)
