from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import os
from keras.models import load_model

app = Flask(__name__, static_url_path='/static')

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 15
CLASSES_LIST = ["NonViolence", "Violence"]

model = load_model("C:\D\Study\Internship\AMZI Smart Solutions\Project\Model.h5")

def preprocess_video(video_path):
    frames_list = []

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0  # Initialize a frame count variable

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Resize the frame to match the model's input size
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize the frame
        frame = frame / 255.0

        frames_list.append(frame)

        frame_count += 1  # Increment frame count

        # Check if we have reached the desired number of frames (SEQUENCE_LENGTH)
        if frame_count == SEQUENCE_LENGTH:
            break  # Exit the loop when we have enough frames

    video_capture.release()

    # Ensure that the number of frames matches SEQUENCE_LENGTH
    while len(frames_list) < SEQUENCE_LENGTH:
        # If we have fewer frames, duplicate the last frame to match the desired length
        frames_list.append(frames_list[-1].copy())

    # Convert the list of frames to a numpy array with shape (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    frames_array = np.array(frames_list)

    # Expand the dimensions to match the expected input shape (None, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    frames_array = np.expand_dims(frames_array, axis=0)

    return frames_array

def predict_video(video_path):
    frames_list = preprocess_video(video_path)
    predictions = []

    for frame in frames_list:
        prediction = model.predict(np.expand_dims(frame, axis=0))
        predictions.append(prediction)

    final_prediction = combine_predictions(predictions)

    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(final_prediction)
    predicted_class = CLASSES_LIST[predicted_class_idx]
    confidence = final_prediction[0][predicted_class_idx]

    return predicted_class, confidence

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
        if 'video' not in request.files:
            return render_template('index.html', error="No video selected for upload")

        video_file = request.files['video']
        
        # Check if the file is empty
        if video_file.filename == '':
            return render_template('index.html', error="No video selected for upload")

        video_path = 'temp_video.mp4'
        video_file.save(video_path)
        predicted_class, confidence = predict_video(video_path)
        return render_template('index.html', prediction=predicted_class, confidence=confidence)
    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
