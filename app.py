import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import cv2
import tensorflow as tf



# Load model
model = load_model("deepfake_detector.h5")

# Load MobileNetV2 for feature extraction
mobilenet = MobileNetV2(include_top=False, input_shape=(224, 224, 3), pooling='avg')
mobilenet.trainable = False

def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    if len(frames) < max_frames:
        frames.extend([frames[-1]] * (max_frames - len(frames)))
    return np.array(frames)

def extract_features(frames):
    frames = tf.keras.applications.mobilenet_v2.preprocess_input(frames.astype(np.float32))
    return mobilenet.predict(frames, verbose=0)

def predict_video(video_path):
    frames = extract_frames(video_path)
    features = extract_features(frames)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return "FAKE VIDEO" if prediction[0][0] > 0.5 else "REAL VIDEO"

st.title("Deepfake Video Detector")
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    result = predict_video(video_path)
    st.write(f"Prediction: **{result}**")
