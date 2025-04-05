import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
IMG_SIZE = 224
SEQ_LEN = 30

# Step 1: Extract frames from video
def extract_frames(video_path, max_frames=SEQ_LEN):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()
    if len(frames) < SEQ_LEN:
        frames.extend([frames[-1]] * (SEQ_LEN - len(frames)))
    return np.array(frames)

# Step 2: Feature extraction using MobileNetV2
def extract_features(frames, model):
    frames = tf.keras.applications.mobilenet_v2.preprocess_input(frames.astype(np.float32))
    return model.predict(frames, verbose=0)

# Step 3: Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Load MobileNetV2
mobilenet = MobileNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
mobilenet.trainable = False

# Step 5: Dataset paths (replace with your own paths)
REAL_VIDEOS_PATH = [r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__walking_and_outside_surprised.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__walking_down_indoor_hall_disgust.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__walking_down_street_outside_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__walking_outside_cafe_disgusted.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__hugging_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__talking_against_wall.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__talking_angry_couch.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__walk_down_hall_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__walking_and_outside_surprised.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__walking_down_indoor_hall_disgust.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\17__walking_down_street_outside_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__secret_conversation.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__talking_against_wall.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__walk_down_hall_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__walking_down_street_outside_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\18__walking_outside_cafe_disgusted.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\19__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\19__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\19__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\19__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\19__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\19__walk_down_hall_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\19__walking_down_street_outside_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\19__walking_outside_cafe_disgusted.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__hugging_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__secret_conversation.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__talking_against_wall.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__talking_angry_couch.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__walk_down_hall_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__walking_and_outside_surprised.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\20__walking_down_indoor_hall_disgust.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\21__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\21__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\21__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\21__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\21__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\21__talking_against_wall.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\21__walking_down_street_outside_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\21__walking_outside_cafe_disgusted.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__walk_down_hall_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__walking_and_outside_surprised.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__walking_down_indoor_hall_disgust.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__walking_down_street_outside_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\22__walking_outside_cafe_disgusted.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\23__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\23__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\23__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\23__secret_conversation.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\23__talking_against_wall.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\23__talking_angry_couch.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\23__walk_down_hall_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\24__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\24__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\24__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\24__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\24__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\24__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\24__walking_down_street_outside_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\24__walking_outside_cafe_disgusted.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\25__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\25__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\25__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\25__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\25__walking_outside_cafe_disgusted.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\26__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\26__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\26__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\26__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\26__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\26__walk_down_hall_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\26__walking_down_street_outside_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\26__walking_outside_cafe_disgusted.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__hugging_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__talking_against_wall.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__talking_angry_couch.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__walk_down_hall_angry.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__walking_and_outside_surprised.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\27__walking_down_indoor_hall_disgust.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\28__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\28__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\28__outside_talking_still_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\28__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\28__secret_conversation.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\28__talking_angry_couch.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__exit_phone_room.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__hugging_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__kitchen_pan.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__kitchen_still.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__outside_talking_pan_laughing.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__podium_speech_happy.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__talking_against_wall.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__walk_down_hall_angry.mp4",]
FAKE_VIDEOS_PATH = [r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__secret_conversation__6UBMLXK3.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__secret_conversation__FW94AIMJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__talking_angry_couch__6UBMLXK3.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__talking_angry_couch__FW94AIMJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__walk_down_hall_angry__FW94AIMJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__walking_and_outside_surprised__6UBMLXK3.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__walking_and_outside_surprised__OTGHOG4Z.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_21__kitchen_pan__03X7CELV.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_21__outside_talking_pan_laughing__03X7CELV.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_21__secret_conversation__03X7CELV.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_21__talking_angry_couch__03X7CELV.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_21__walk_down_hall_angry__03X7CELV.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_26__exit_phone_room__BTVMWLG6.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_26__meeting_serious__ZMEJ535O.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_26__outside_talking_pan_laughing__ZMEJ535O.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_26__walking_outside_cafe_disgusted__BTVMWLG6.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_27__hugging_happy__ZYCZ30C0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_27__meeting_serious__ZYCZ30C0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_27__outside_talking_pan_laughing__S2YCUY48.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_27__outside_talking_still_laughing__S2YCUY48.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_27__outside_talking_still_laughing__ZYCZ30C0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_27__talking_against_wall__ZYCZ30C0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_27__talking_angry_couch__ZYCZ30C0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_27__walking_outside_cafe_disgusted__ZYCZ30C0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__exit_phone_room__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__hugging_happy__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__meeting_serious__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__outside_talking_still_laughing__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__secret_conversation__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__talking_against_wall__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__talking_angry_couch__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__walk_down_hall_angry__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__walking_and_outside_surprised__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__walking_down_indoor_hall_disgust__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_01__walking_outside_cafe_disgusted__YVGY8LOK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_03__kitchen_still__ZESUJMM7.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_03__podium_speech_happy__QH3Y0IG0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_03__walking_and_outside_surprised__GYX5OFTD.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_03__walking_down_street_outside_angry__QH3Y0IG0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_03__walking_outside_cafe_disgusted__GYX5OFTD.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_03__walking_outside_cafe_disgusted__QH3Y0IG0.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__exit_phone_room__SUIZOXAJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__hugging_happy__SUIZOXAJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__outside_talking_pan_laughing__8CH7R4LW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__outside_talking_still_laughing__8CH7R4LW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__podium_speech_happy__8CH7R4LW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__secret_conversation__8CH7R4LW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__walk_down_hall_angry__8CH7R4LW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__walk_down_hall_angry__SUIZOXAJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_04__walking_and_outside_surprised__SUIZOXAJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__exit_phone_room__3J3BHSHI.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__exit_phone_room__HZAXM70B.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__kitchen_pan__0M6JNS5D.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__kitchen_pan__3J3BHSHI.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__kitchen_pan__O9WOC1KJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__kitchen_still__HZAXM70B.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__meeting_serious__0M6JNS5D.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__outside_talking_pan_laughing__N8OSN8P6.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__podium_speech_happy__N8OSN8P6.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__podium_speech_happy__SU4OQCS9.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__secret_conversation__O9WOC1KJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__talking_angry_couch__GH8TGTBS.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__talking_angry_couch__J1W9R0NG.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__talking_angry_couch__MKZTXQ2T.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walk_down_hall_angry__0M6JNS5D.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walk_down_hall_angry__MKZTXQ2T.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_and_outside_surprised__N8OSN8P6.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_and_outside_surprised__O9WOC1KJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_down_indoor_hall_disgust__GH8TGTBS.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_down_indoor_hall_disgust__HZAXM70B.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_down_indoor_hall_disgust__SU4OQCS9.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_down_indoor_hall_disgust__U6MDWIHG.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_down_street_outside_angry__MKZTXQ2T.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_outside_cafe_disgusted__37DH75GQ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_06__walking_outside_cafe_disgusted__GH8TGTBS.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__exit_phone_room__1JCLEEBQ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__exit_phone_room__9NVRE2KQ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__hugging_happy__1JCLEEBQ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__hugging_happy__O4SXNLRL.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__kitchen_pan__34PVT42V.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__kitchen_pan__O4SXNLRL.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__kitchen_still__1ZE4HC06.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__kitchen_still__TBLYQBIY.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__meeting_serious__1H07DFQJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__meeting_serious__1JCLEEBQ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__meeting_serious__O4SXNLRL.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__outside_talking_pan_laughing__34PVT42V.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__outside_talking_pan_laughing__O4SXNLRL.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__podium_speech_happy__0IYV5DQ5.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__podium_speech_happy__1KPVZAAP.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__podium_speech_happy__UH60ROEK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__secret_conversation__0IYV5DQ5.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__secret_conversation__1JCLEEBQ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__secret_conversation__1VMZUH1W.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__talking_against_wall__1ZE4HC06.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__talking_against_wall__O4SXNLRL.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__talking_angry_couch__1VMZUH1W.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__walk_down_hall_angry__1JCLEEBQ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__walk_down_hall_angry__O4SXNLRL.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__walk_down_hall_angry__U7DEOZNV.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__walking_and_outside_surprised__1VMZUH1W.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__walking_and_outside_surprised__UH60ROEK.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__walking_down_street_outside_angry__O4SXNLRL.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_07__walking_outside_cafe_disgusted__1H07DFQJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__exit_phone_room__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__exit_phone_room__HIH8YA82.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__hugging_happy__9TDCEK1Q.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__hugging_happy__HIH8YA82.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__kitchen_pan__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__kitchen_pan__9TDCEK1Q.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__kitchen_pan__HIH8YA82.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__kitchen_still__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__meeting_serious__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__meeting_serious__9TDCEK1Q.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__outside_talking_pan_laughing__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__outside_talking_pan_laughing__HIH8YA82.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__outside_talking_still_laughing__9TDCEK1Q.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__podium_speech_happy__9TDCEK1Q.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__podium_speech_happy__HIH8YA82.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__secret_conversation__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__talking_angry_couch__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__walk_down_hall_angry__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__walk_down_hall_angry__9TDCEK1Q.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__walking_down_street_outside_angry__6KUOFMZW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__walking_down_street_outside_angry__9TDCEK1Q.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_09__walking_down_street_outside_angry__HIH8YA82.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_11__meeting_serious__NZJ1YEWE.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_12__hugging_happy__9D2ZHEKW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_12__outside_talking_pan_laughing__9D2ZHEKW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_12__podium_speech_happy__9D2ZHEKW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_12__walking_and_outside_surprised__9D2ZHEKW.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__exit_phone_room__CP5HFV3K.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__exit_phone_room__PLNVLO74.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__hugging_happy__PLNVLO74.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__kitchen_pan__0RKOCC6A.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__kitchen_pan__2YSYT2N3.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__kitchen_pan__CP5HFV3K.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__kitchen_still__2YSYT2N3.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__meeting_serious__CP5HFV3K.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\02_13__outside_talking_pan_laughing__0RKOCC6A.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__kitchen_still__D8GWGO2A.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__kitchen_still__FW94AIMJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__meeting_serious__6UBMLXK3.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__meeting_serious__D8GWGO2A.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__meeting_serious__FW94AIMJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__outside_talking_pan_laughing__6UBMLXK3.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__outside_talking_pan_laughing__OTGHOG4Z.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__outside_talking_still_laughing__6UBMLXK3.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__outside_talking_still_laughing__FW94AIMJ.mp4",
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__outside_talking_still_laughing__OTGHOG4Z.mp4",
]

# Step 6: Prepare dataset
X = []
y = []

print("Processing real videos...")
for video_path in tqdm(REAL_VIDEOS_PATH):
    frames = extract_frames(video_path)
    features = extract_features(frames, mobilenet)
    X.append(features)
    y.append(0)

print("Processing fake videos...")
for video_path in tqdm(FAKE_VIDEOS_PATH):
    frames = extract_frames(video_path)
    features = extract_features(frames, mobilenet)
    X.append(features)
    y.append(1)

X = np.array(X)
y = np.array(y)

# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train model
model = build_model((SEQ_LEN, 1280))  # MobileNetV2 outputs 1280 features
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=4)

# Step 9: Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# =============================================================================
# Save trained model
model.save("deepfake_detector.h5")
print("✅ Model saved successfully!")

# ================================================================================
# Load the trained model
from tensorflow.keras.models import load_model

model = load_model("deepfake_detector.h5")
print("✅ Model loaded successfully!")
# ==========================================================================================
def predict_video(video_path, model, mobilenet):
    print(f"Processing video: {video_path}")

    # Extract frames
    frames = extract_frames(video_path)
    
    # Extract features
    features = extract_features(frames, mobilenet)
    
    # Reshape for LSTM input
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(features)
    
    # Convert to label
    label = "FAKE" if prediction[0][0] > 0.5 else "REAL"
    
    print(f"🧐 Prediction: {label} ({prediction[0][0]:.2f})")
    return label
new_video = r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__exit_phone_room.mp4"
predict_video(new_video, model, mobilenet)
