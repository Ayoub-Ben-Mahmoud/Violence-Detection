from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import os
from keras.models import load_model

app = Flask(__name__)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 10
CLASSES_LIST = ["NonViolence", "Violence"]

model = load_model("C:\D\Study\Internship\AMZI Smart Solutions\Project\Model.h5")

def preprocess_video(video_path):
    frames_list = []

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Resize the frame to match the model's input size
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize the frame
        frame = frame / 255.0

        frames_list.append(frame)

    video_capture.release()

    # Ensure that the number of frames matches SEQUENCE_LENGTH
    if len(frames_list) != SEQUENCE_LENGTH:
        # Handle this case as needed, e.g., by duplicating frames or skipping the video
        pass

    return frames_list

def predict_video(video_path):
    frames_list = preprocess_video(video_path)
    predictions = []

    for frame in frames_list:
        prediction = model.predict(np.expand_dims(frame, axis=0))
        predictions.append(prediction)

    final_prediction = combine_predictions(predictions)

    return final_prediction

def combine_predictions(predictions):
    # Combine predictions if necessary (e.g., averaging)
    averaged_prediction = np.mean(predictions, axis=0)
    return averaged_prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        video_file = request.files['video']
        video_path = 'temp_video.mp4'
        video_file.save(video_path)
        prediction = predict_video(video_path)
        class_name = CLASSES_LIST[np.argmax(prediction)]
        return render_template('index.html', prediction=class_name)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
