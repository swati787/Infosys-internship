import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Loading
model = tf.keras.models.load_model("../activity_model.h5")

# Classes
categories = ["walking", "running", "cycling"]

st.title("AI-Based Intelligent Video Surveillance System")
st.write("Upload an image to detect activity")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Prediction
    prediction = model.predict(img_input)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display image
    st.image(img, caption="Uploaded Image", channels="BGR")

    # Result display
    if predicted_class in ["walking", "running"]:
        st.success(f"🟢 Authorized Activity: {predicted_class.upper()} ({confidence:.2f}%)")
    else:
        st.error(f"🔴 Unauthorized Activity: {predicted_class.upper()} ({confidence:.2f}%)")
