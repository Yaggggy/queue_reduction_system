from flask import Flask, render_template, Response, jsonify
import cv2
import os
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load known faces and names from Training_images folder
path = 'Training_images'
images = []
classNames = []
image_list = os.listdir(path)

for image_name in image_list:
    cur_img = cv2.imread(f'{path}/{image_name}')
    images.append(cur_img)
    classNames.append(os.path.splitext(image_name)[0])

# Function to encode faces
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

known_encodings = find_encodings(images)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (name TEXT, login_time TEXT)''')
    conn.commit()
    conn.close()

# Log the name and time into the database
def log_to_db(name):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    login_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO users (name, login_time) VALUES (?, ?)", (name, login_time))
    conn.commit()
    conn.close()

# Keep track of logged people within the session
logged_people = set()

# Face recognition and detection
def gen_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            faces_in_frame = face_recognition.face_locations(rgb_small_frame)
            encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, faces_in_frame)

            for encoding, face_location in zip(encodings_in_frame, faces_in_frame):
                matches = face_recognition.compare_faces(known_encodings, encoding)
                face_distance = face_recognition.face_distance(known_encodings, encoding)
                match_index = np.argmin(face_distance)

                if matches[match_index]:
                    name = classNames[match_index].upper()

                    # Log to database if the person hasn't been logged already
                    if name not in logged_people:
                        log_to_db(name)
                        logged_people.add(name)
                else:
                    name = "Unknown"

                # Draw rectangle around face
                top, right, bottom, left = face_location
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
