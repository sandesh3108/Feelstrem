import cv2
import numpy as np
import time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = load_model('main_emotion_detection_model.h5')

# Define emotion labels (based on your training data)
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# Load song recommendations from CSV
song_data = pd.read_csv('data_moods.csv')
song_recommendations = {}
for _, row in song_data.iterrows():
    emotion = row['mood']
    song_info = f"{row['name']} - {row['artist']} - {row['album']}"

    if emotion in song_recommendations:
        song_recommendations[emotion].append(song_info)
    else:
        song_recommendations[emotion] = [song_info]

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Initialize variables to track time and store detected emotions
last_detected_time = time.time()
detected_emotions = []  # List to store all detected emotions
detection_interval = 10  # seconds

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    current_time = time.time()

    if current_time - last_detected_time >= detection_interval:
        # Reset the detection time
        last_detected_time = current_time

        for (x, y, w, h) in faces:
            # Extract the region of interest (face)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0

            # Convert the grayscale image to a 3-channel image
            roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            roi_array = img_to_array(roi_color)
            roi_array = np.expand_dims(roi_array, axis=0)

            # Predict the emotion
            predictions = model.predict(roi_array)
            print(f"Predictions: {predictions}")  # Debugging line
            max_index = np.argmax(predictions[0])
            emotion_label = emotion_labels[max_index]

            # Append the detected emotion to the list
            detected_emotions.append(emotion_label)

            # Recommend songs based on the detected emotion
            recommended_songs = song_recommendations.get(emotion_label, [])
            print(f"Detected Emotion: {emotion_label}")  # Debugging line
            print(f"Recommended Songs: {recommended_songs}")  # Debugging line

            if recommended_songs:
                song_list = ", ".join(recommended_songs)
                emotion_info = f"{emotion_label}: {song_list}"
            else:
                emotion_info = emotion_label

            # Draw a rectangle around the face and put the emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the emotion label
            cv2.putText(frame, emotion_label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Display the song recommendations
            cv2.putText(frame, emotion_info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 1)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
