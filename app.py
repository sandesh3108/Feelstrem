from flask import Flask, jsonify, render_template
from emotion_recognition import EmotionRecognizer
import threading
import time

app = Flask(__name__)

# Initialize EmotionRecognizer
emotion_recognizer = EmotionRecognizer('main_emotion_detection_model.h5', 'data_moods.csv')

# Global variables to store the current emotion and recommended songs
current_emotion = None
current_songs = []

# Function to update emotion every 1 minute
def update_emotion():
    global current_emotion, current_songs
    while True:
        detected_emotion, recommended_songs = emotion_recognizer.recognize_emotion()
        current_emotion = detected_emotion
        current_songs = recommended_songs
        time.sleep(60)  # Wait for 1 minute before detecting the next emotion

# Start the emotion detection thread
emotion_thread = threading.Thread(target=update_emotion)
emotion_thread.daemon = True
emotion_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_emotion')
def get_emotion():
    return jsonify({'emotion': current_emotion, 'recommended_songs': current_songs})

if __name__ == '__main__':
    app.run(debug=True)
