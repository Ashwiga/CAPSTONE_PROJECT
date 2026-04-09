import cv2
import numpy as np
import pickle
import csv
import os
import time
import hashlib
import queue
import threading
import requests
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

import psycopg2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import config
from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()


# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# Store live bus location from browser
current_lat = None
current_lon = None


# -----------------------------
# TWILIO CONFIG
# -----------------------------
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")


twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)


# -----------------------------
# GET REAL BUS LOCATION
# -----------------------------
def get_bus_location():

    global current_lat, current_lon

    if current_lat and current_lon:

        return f"https://www.google.com/maps?q={current_lat},{current_lon}"

    return "Location unavailable"


# -----------------------------
# Camera Control
# -----------------------------
camera_active = False
cap = None
last_detection_time = "No detection yet"


# -----------------------------
# Load Models
# -----------------------------
yolo_model = YOLO("models/yolov8n-face.pt")
embedder = FaceNet()


# -----------------------------
# Load Face Embeddings
# -----------------------------
with open("embeddings/faces.pkl", "rb") as f:
    faces_db = pickle.load(f)

print("Loaded faces.pkl successfully")


# -----------------------------
# SETTINGS
# -----------------------------
THRESHOLD = 0.78
PADDING = 25
COOLDOWN_SECONDS = 8

attendance_file = "attendance.csv"

last_seen_time = {}
last_event = {}
recent_logs = []


# -----------------------------
# Database Queue
# -----------------------------
attendance_queue = queue.Queue()


# -----------------------------
# Create CSV
# -----------------------------
if not os.path.exists(attendance_file):

    with open(attendance_file, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Event", "Confidence", "Hash"])


# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_db_connection():

    try:

        return psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)

    except Exception as e:

        print("DB connection failed:", e)
        return None


# -----------------------------
# DB Worker
# -----------------------------
def db_worker():

    while True:

        data = attendance_queue.get()

        if data is None:
            break

        name, date_str, time_str, event_type, confidence, data_hash = data

        save_attendance_to_db(
            name,
            date_str,
            time_str,
            event_type,
            confidence,
            data_hash
        )

        attendance_queue.task_done()


db_thread = threading.Thread(target=db_worker, daemon=True)
db_thread.start()


# -----------------------------
# EMAIL FUNCTION
# -----------------------------
def send_email(to_email, subject, body):

    try:

        msg = MIMEMultipart()

        msg["From"] = config.SENDER_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)
        server.starttls()

        server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)

        server.sendmail(
            config.SENDER_EMAIL,
            to_email,
            msg.as_string()
        )

        server.quit()

        print("Email sent")

    except Exception as e:

        print("Email failed:", e)


# -----------------------------
# SMS FUNCTION
# -----------------------------
def send_sms(to_number, message):

    try:

        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=to_number
        )

        print("SMS sent to", to_number)

    except Exception as e:

        print("SMS failed:", e)


# -----------------------------
# Parent Contacts
# -----------------------------
parent_emails = {

    "Ashwiga": "22cs021@kpriet.ac.in",
    "Jagan Kumar": "22ec039@kpriet.ac.in",
    "Pavithra": "22am042@kpriet.ac.in"

}

parent_numbers = {

    "Ashwiga": "+919342310715",
    "Jagan Kumar": "+919360868375",
    "Pavithra": "+919385632014"

}


# -----------------------------
# Face Recognition
# -----------------------------
def recognize_face(face_img):

    face_img = cv2.resize(face_img, (160, 160))
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    emb = embedder.embeddings([face_rgb])[0]

    best_name = "Unknown"
    best_score = -1

    for person_name, embeddings_list in faces_db.items():

        if len(embeddings_list) == 0:
            continue

        sims = cosine_similarity([emb], embeddings_list)[0]
        max_sim = float(np.max(sims))

        if max_sim > best_score:

            best_score = max_sim
            best_name = person_name

    if best_score < THRESHOLD:

        return "Unknown", best_score

    return best_name, best_score


# -----------------------------
# Save Attendance to DB
# -----------------------------
def save_attendance_to_db(
        name,
        date_str,
        time_str,
        event_type,
        confidence,
        data_hash
):

    conn = get_db_connection()

    if conn:

        try:

            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO attendance 
                (name,date,time,event,confidence,hash)
                VALUES (%s,%s,%s,%s,%s,%s)
                """,
                (
                    name,
                    date_str,
                    time_str,
                    event_type,
                    float(confidence),
                    data_hash
                )
            )

            conn.commit()

            cur.close()
            conn.close()

        except Exception as e:

            print("DB save failed:", e)


# -----------------------------
# Mark Attendance
# -----------------------------
def mark_attendance(name, confidence):

    global last_detection_time

    now = datetime.now()

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    location_link = get_bus_location()

    if name not in last_event:

        event_type = "Board"

    else:

        event_type = "Deboard" if last_event[name] == "Board" else "Board"

    data_hash = hashlib.md5(
        f"{name}{date_str}{time_str}{event_type}".encode()
    ).hexdigest()[:8]


    # CSV
    with open(attendance_file, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            name,
            date_str,
            time_str,
            event_type,
            float(confidence),
            data_hash
        ])


    # Queue DB
    attendance_queue.put(
        (name, date_str, time_str, event_type, confidence, data_hash)
    )


    # EMAIL ALERT
    if name in parent_emails:

        email = parent_emails[name]

        subject = f"Bus Alert - {event_type}"

        body = f"""
Hello Parent,

{name} has {event_type}ed the bus.

Date : {date_str}
Time : {time_str}

Location
{location_link}
"""

        send_email(email, subject, body)


    # SMS ALERT
    if name in parent_numbers:

        phone = parent_numbers[name]

        sms = f"""
Bus Alert

{name} has {event_type}ed the bus
Time: {time_str}

Location
{location_link}
"""

        send_sms(phone, sms)


    last_event[name] = event_type
    last_detection_time = time_str

    recent_logs.append({

        "name": name,
        "date": date_str,
        "time": time_str,
        "event": event_type

    })

    if len(recent_logs) > 20:
        recent_logs.pop(0)


# -----------------------------
# Video Stream
# -----------------------------
def generate_frames():

    global camera_active, cap

    while True:

        if not camera_active:

            if cap:

                cap.release()
                cap = None

            time.sleep(0.3)
            continue

        if cap is None:

            cap = cv2.VideoCapture(0)

        success, frame = cap.read()

        if not success:
            continue

        results = yolo_model(frame)

        for box in results[0].boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(frame.shape[1], x2 + PADDING)
            y2 = min(frame.shape[0], y2 + PADDING)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            name, score = recognize_face(face_crop)

            if name != "Unknown":

                current_time = time.time()

                if name not in last_seen_time or \
                        (current_time - last_seen_time[name]) > COOLDOWN_SECONDS:

                    mark_attendance(name, score)

                    last_seen_time[name] = current_time

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{name} ({score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame +
               b"\r\n")


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/start_camera")
def start_camera():

    global camera_active
    camera_active = True

    return jsonify({"status": "Camera Started"})


@app.route("/stop_camera")
def stop_camera():

    global camera_active, cap

    camera_active = False

    if cap:

        cap.release()
        cap = None

    return jsonify({"status": "Camera Stopped"})


@app.route("/logs")
def logs():
    return jsonify(recent_logs[::-1])

@app.route("/update_location", methods=["POST"])
def update_location():

    global current_lat, current_lon

    data = request.get_json()

    current_lat = data["latitude"]
    current_lon = data["longitude"]

    return jsonify({"status": "location updated"})


@app.route("/system_status")
def system_status():

    return jsonify({

        "camera": "Active" if camera_active else "Stopped",
        "database": "Connected" if get_db_connection() else "Failed",
        "last_detection": last_detection_time

    })


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)