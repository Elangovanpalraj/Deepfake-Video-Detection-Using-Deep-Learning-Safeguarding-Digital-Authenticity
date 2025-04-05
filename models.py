import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tqdm import tqdm

# Use forward slashes in Windows paths
REAL_VIDEOS_PATH = r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__walking_and_outside_surprised.mp4",
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
r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\real_videos\16__walk_down_hall_angry.mp4",
FAKE_VIDEOS_PATH = r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__secret_conversation__6UBMLXK3.mp4",
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

FRAME_SIZE = (224, 224)
SEQUENCE_LENGTH = 10

# Load YOLOv5 model for face detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_faces(frame):
    """Detect faces in a frame using YOLOv5"""
    results = yolo_model(frame)
    for *xyxy, conf, cls in results.xyxy[0]:
        if int(cls) == 0 and conf > 0.5:  # Class 0 is "person"
            x1, y1, x2, y2 = map(int, xyxy)
            return frame[y1:y2, x1:x2]
    return None

def extract_frames(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract face-only frames from a video"""
    if not os.path.exists(video_path):
        print(f"⚠️ Warning: Video '{video_path}' not found. Skipping...")
        return None

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        face = detect_faces(frame)
        if face is not None:
            face = cv2.resize(face, FRAME_SIZE)
            frames.append(face)
            frame_count += 1

    cap.release()
    return np.array(frames) if len(frames) == max_frames else None

# Load dataset safely
X, y = [], []
for folder, label in [(REAL_VIDEOS_PATH, 0), (FAKE_VIDEOS_PATH, 1)]:
    if not os.path.exists(folder):
        print(f"❌ Error: Folder '{folder}' does not exist! Check the path.")
        continue  # Skip missing folder

    filenames = os.listdir(folder)
    if not filenames:
        print(f"⚠️ Warning: No videos found in '{folder}'. Skipping...")
        continue  # Skip empty folder

    for filename in tqdm(filenames, desc=f"Processing {folder}"):
        video_path = os.path.join(folder, filename)
        frames = extract_frames(video_path)
        if frames is not None:
            X.append(frames)
            y.append(label)

if len(X) == 0:
    print("❌ No valid videos found. Please check dataset paths and video files.")
    exit()

X = np.array(X) / 255.0  # Normalize
y = np.array(y)

# Build Efficient CNN-LSTM Model
cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

model = Sequential([
    TimeDistributed(cnn, input_shape=(SEQUENCE_LENGTH, 224, 224, 3)),
    LSTM(128, return_sequences=False),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
model.fit(X, y, batch_size=16, epochs=5, validation_split=0.2)

# Save Model
model.save("optimized_deepfake_detector.h5")
print("✅ Model Trained & Saved!")
