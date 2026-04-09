# ===================== SYSTEM CONFIG =====================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ===================== IMPORTS =====================
import streamlit as st
import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, date
import psycopg2
import pandas as pd
import hashlib

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Smart Face Attendance",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Smart Face Recognition Attendance System")

# ===================== CONSTANTS =====================
EMBEDDING_FILE = "embeddings/faces.pkl"
THRESHOLD = 0.7
EVENT_TYPE = "Board"
LOCATION = "School Bus"

# ===================== DATABASE CONFIG =====================
DB_CONFIG = {
    "host": "localhost",
    "dbname": "attendance_db",
    "user": "postgres",
    "password": "postgres",
    "port": 5432
}

# ===================== DATABASE FUNCTIONS =====================
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def create_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            student_id INTEGER,
            date DATE NOT NULL,
            time TIME NOT NULL,
            event VARCHAR(20) NOT NULL
                CHECK (event IN ('Board', 'Deboard', 'ABSENT')),
            hash VARCHAR(20) NOT NULL,
            location VARCHAR(100),
            confidence_score DECIMAL(5,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_attendance(name, confidence):
    conn = get_db_connection()
    cur = conn.cursor()

    today = date.today()
    now_time = datetime.now().time()

    # 🔑 Convert numpy.float32 → Python float
    confidence = float(confidence)

    hash_input = f"{name}{today}{EVENT_TYPE}".encode()
    data_hash = hashlib.sha1(hash_input).hexdigest()[:20]

    try:
        cur.execute("""
            INSERT INTO attendance (
                name,
                date,
                time,
                event,
                hash,
                location,
                confidence_score
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            name,
            today,
            now_time,
            EVENT_TYPE,
            data_hash,
            LOCATION,
            round(confidence * 100, 2)
        ))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def load_today_attendance():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT name, time, event, confidence_score
        FROM attendance
        WHERE date = CURRENT_DATE
        ORDER BY time
    """, conn)
    conn.close()
    return df

# ===================== INIT DB =====================
create_table()

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_models():
    return MTCNN(), FaceNet()

detector, embedder = load_models()

# ===================== LOAD FACE DATABASE =====================
def load_database():
    if not os.path.exists(EMBEDDING_FILE):
        return {}
    with open(EMBEDDING_FILE, "rb") as f:
        return pickle.load(f)

database = load_database()

# ===================== FACE EMBEDDING =====================
def get_face_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]

# ===================== SAFE FACE DETECTION =====================
def safe_detect_faces(rgb_img):
    try:
        faces = detector.detect_faces(rgb_img)
        return faces if faces else []
    except Exception:
        return []

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("📊 Control Panel")
    st.write(f"👤 Registered Faces: {len(database)}")
    run_camera = st.checkbox("▶ Start Camera")
    st.markdown(f"**📌 Event:** {EVENT_TYPE}")

# ===================== CAMERA LOOP =====================
frame_box = st.empty()
marked = set()

if run_camera:
    cap = cv2.VideoCapture(0)

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = safe_detect_faces(rgb)

        for face in faces:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            if w < 70 or h < 70:
                continue

            face_img = rgb[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            name = "Unknown"
            confidence = 0.0

            if database:
                emb = get_face_embedding(face_img)
                sims = {
                    person: cosine_similarity(
                        emb.reshape(1, -1),
                        db_emb.reshape(1, -1)
                    )[0][0]
                    for person, db_emb in database.items()
                }

                best_match = max(sims, key=sims.get)
                confidence = float(sims[best_match])  # 🔑 FIX HERE

                if confidence >= THRESHOLD:
                    name = best_match
                    if name not in marked:
                        save_attendance(name, confidence)
                        marked.add(name)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                f"{name} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2
            )

        frame_box.image(frame, channels="BGR")

    cap.release()

# ===================== ATTENDANCE DISPLAY =====================
st.divider()
st.subheader("📋 Today's Attendance")

df = load_today_attendance()
if not df.empty:
    st.dataframe(df, use_container_width=True)
else:
    st.info("No attendance marked yet.")
