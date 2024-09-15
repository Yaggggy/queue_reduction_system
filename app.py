from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime
import dlib
from scipy.spatial import distance as dist
import csv

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

# SQLite initialization
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (name TEXT, login_time TEXT)''')
    conn.commit()
    conn.close()

def log_to_db(name):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    login_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO users (name, login_time) VALUES (?, ?)", (name, login_time))
    conn.commit()
    conn.close()

# CSV file names
csv_file1 = 'reduction\logfile1.csv'
csv_file2 = 'reduction\logfile2.csv'

# Check if the CSV files exist, if not, create them and add headers
def init_csv_files():
    if not os.path.exists(csv_file1):
        with open(csv_file1, mode='w', newline='') as file1:
            writer = csv.writer(file1)
            writer.writerow(["Name", "Login Time"])
    
    if not os.path.exists(csv_file2):
        with open(csv_file2, mode='w', newline='') as file2:
            writer = csv.writer(file2)
            writer.writerow(["Name", "Login Time"])

# Function to log the name and time to two CSV files
def log_to_csv_files(name):
    login_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Log to first CSV file
    with open(csv_file1, mode='a', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerow([name, login_time])
    
    # Log to second CSV file
    with open(csv_file2, mode='a', newline='') as file2:
        writer = csv.writer(file2)
        writer.writerow([name, login_time])

# Function to log user into both DB and CSV files
def log_user(name):
    log_to_db(name)
    log_to_csv_files(name)

# Blink detection configuration
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('reduction\shape_predictor_68_face_landmarks.dat')

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(landmarks, frame):
    (lStart, lEnd) = (42, 48)  # Left eye indices
    (rStart, rEnd) = (36, 42)  # Right eye indices

    left_eye = landmarks[lStart:lEnd]
    right_eye = landmarks[rStart:rEnd]

    # Visualize the eyes for debugging
    for (x, y) in left_eye:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    for (x, y) in right_eye:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    leftEAR = eye_aspect_ratio(left_eye)
    rightEAR = eye_aspect_ratio(right_eye)
    ear = (leftEAR + rightEAR) / 2.0

    # Debug print the EAR value to see if blink is detected
    print(f"EAR: {ear}")

    # Blink detected if EAR is less than a threshold (tune if needed)
    return ear < 0.29  # Adjust this threshold for strictness

# Thresholds for confirming a valid user
VALIDATION_FRAMES_THRESHOLD = 5  # Number of consecutive frames required
current_frame_count = 0            # Track consecutive valid frames
current_name = None                # Track currently detected person
current_status = "Unknown"         # Status: "Valid" or "Fake"
logged_people = set()

# Face recognition and detection
def gen_frames():
    global current_frame_count, current_name, current_status
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

                    # Detect liveness through blink detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rects = detector(gray, 0)

                    is_liveness_confirmed = False
                    for rect in rects:
                        shape = predictor(gray, rect)
                        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

                        # Check blink detection and pass the frame for visualization
                        if detect_blink(landmarks, frame):
                            is_liveness_confirmed = True

                    if is_liveness_confirmed and name == current_name:
                        current_frame_count += 1
                        current_status = "Valid"
                    else:
                        current_frame_count = 0
                        current_name = name
                        current_status = "Fake"  # If liveness fails, mark as "Fake"

                    # Check if the user has been valid for enough frames
                    if current_frame_count >= VALIDATION_FRAMES_THRESHOLD:
                        if name not in logged_people:
                            log_user(name)
                            logged_people.add(name)
                            current_frame_count = 0  # Reset counter after logging

                else:
                    name = "Unknown"
                    current_frame_count = 0  # Reset counter for unknown faces
                    current_status = "Unknown"

                # Draw rectangle around face
                top, right, bottom, left = face_location
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Display name and status ("Valid" or "Fake")
                display_text = f"{name} - {current_status}"
                cv2.putText(frame, display_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
    init_csv_files()
    init_db()
    app.run(debug=True)
