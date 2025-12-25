


# AI-Based Intelligent Video Surveillance System for Activity Recognition

This project implements an AI-based intelligent surveillance system to classify human activities in park environments as **authorized** or **unauthorized** using deep learning techniques.

The system is developed using a **Convolutional Neural Network (CNN)** trained on image frames extracted from surveillance videos. A **Streamlit-based web interface** is used to visualize predictions with clear color-coded alerts.

---

##  Project Features

- Image-based activity recognition using CNN
- Classification of activities:
  - Walking – Authorized
  - Running – Authorized
  - Cycling – Unauthorized
- Confidence score based on softmax probability
- Color-coded output:
  - Green → Authorized activity
  - Red → Unauthorized activity
- Interactive Streamlit web dashboard

---

##  Project Workflow

1. Collected surveillance videos from park scenarios  
2. Extracted image frames using OpenCV  
3. Organized data into activity-based folders  
4. Split dataset into training and testing sets  
5. Trained a CNN model for activity classification  
6. Tested predictions on new images  
7. Built a Streamlit UI for user interaction  

---

##  Tech Stack

- **Programming Language**: Python  
- **Libraries & Frameworks**:
  - TensorFlow / Keras
  - OpenCV
  - NumPy
  - Streamlit  
- **Model Type**: Convolutional Neural Network (CNN)

---


