'''
def predict_video(video_path, model, mobilenet):
    frames = extract_frames(video_path)
    features = extract_features(frames, mobilenet)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    prediction = model.predict(features)
    return "FAKE" if prediction > 0.5 else "REAL"

# Example usage
video_path = r"D:\STET-details\MCA\Nivedha\Deepfake Video Detection\nivethan\fake_videos\01_20__kitchen_still__D8GWGO2A.mp4"  # Change this to your test video
result = predict_video(video_path, model, mobilenet)
print(f"🔍 Prediction: {result}")
'''

