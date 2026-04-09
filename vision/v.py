import cv2
import numpy as np
import pickle
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load YOLO + FaceNet
# -----------------------------
yolo_model = YOLO("models/yolov8n-face.pt")   # change if needed
embedder = FaceNet()

# -----------------------------
# Load Embeddings Database
# -----------------------------
with open("embeddings/faces.pkl", "rb") as f:
    faces_db = pickle.load(f)

print("✅ Loaded faces.pkl successfully")
for name in faces_db:
    print(name, "->", len(faces_db[name]), "embeddings")

# -----------------------------
# Settings
# -----------------------------
THRESHOLD = 0.78   # important! higher means stricter
PADDING = 25

attendance_file = "attendance.csv"
attendance_marked = set()

# -----------------------------
# Create CSV if not exists
# -----------------------------
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time", "Status"])

# -----------------------------
# Helper: Mark Attendance
# -----------------------------
def mark_attendance(name, status="Boarded"):
    if name in attendance_marked:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now, status])

    attendance_marked.add(name)
    print(f"✅ Attendance marked: {name} ({status})")

# -----------------------------
# Helper: Recognize Face
# -----------------------------
def recognize_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    emb = embedder.embeddings([face_rgb])[0]

    best_name = "Unknown"
    best_score = -1

    # Compare with all embeddings of each person
    for person_name, embeddings_list in faces_db.items():
        if len(embeddings_list) == 0:
            continue

        # Compute cosine similarity for all embeddings
        sims = cosine_similarity([emb], embeddings_list)[0]
        max_sim = np.max(sims)

        if max_sim > best_score:
            best_score = max_sim
            best_name = person_name

    # Threshold check
    if best_score < THRESHOLD:
        return "Unknown", best_score
    else:
        return best_name, best_score

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

print("\n🎥 Camera started... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    for box in results[0].boxes:
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Add padding
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(frame.shape[1], x2 + PADDING)
        y2 = min(frame.shape[0], y2 + PADDING)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Recognize
        name, score = recognize_face(face_crop)

        # Draw box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} ({score:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Attendance only if recognized
        if name != "Unknown":
            mark_attendance(name, "Boarded")

    cv2.imshow("YOLO + FaceNet Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Program closed")
