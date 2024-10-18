import cv2
import numpy as np
import pandas as pd
import random
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class EmotionRecognizer:
    def __init__(self, model_path, csv_path):
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
        self.song_recommendations = self.load_songs(csv_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_songs(self, csv_path):
        """Load song recommendations from a CSV file."""
        song_data = pd.read_csv(csv_path)
        recommendations = {}
        for _, row in song_data.iterrows():
            emotion = row['mood']
            song_info = f"{row['name']} - {row['artist']} - {row['album']}"
            recommendations.setdefault(emotion, []).append(song_info)
        return recommendations

    def recognize_emotion(self, detection_duration=10):
        """Open camera, detect emotion for a given time, then return detected emotion and random songs."""
        cap = cv2.VideoCapture(0)  # Open the camera
        start_time = time.time()  # Start time to limit detection duration

        detected_emotion = None
        recommended_songs = []

        while time.time() - start_time < detection_duration:
            ret, frame = cap.read()  # Read the camera frame
            if not ret:
                break  # If no frame is captured, exit

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                # Process the first face detected
                (x, y, w, h) = faces[0]
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                roi_array = img_to_array(roi_color)
                roi_array = np.expand_dims(roi_array, axis=0)

                # Predict emotion
                predictions = self.model.predict(roi_array)
                max_index = np.argmax(predictions[0])
                detected_emotion = self.emotion_labels[max_index]

                # Shuffle and select 10 random songs for the detected emotion
                songs_for_emotion = self.song_recommendations.get(detected_emotion, [])
                random.shuffle(songs_for_emotion)
                recommended_songs = songs_for_emotion[:10]

                # Display emotion on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, detected_emotion, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show the camera frame
            cv2.imshow('Emotion Detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()  # Release the camera resource
        cv2.destroyAllWindows()  # Close the display window

        return detected_emotion, recommended_songs
