import cv2
import os
import json
import numpy as np

DATASET_DIR = "dataset"
OUTPUT_DIR = "embeddings"

MODEL_PATH = os.path.join(OUTPUT_DIR, "faces.yml")
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, "label_map.json")

MAX_FACES_PER_LABEL = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("❌ Haar cascade not loaded")

# LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

faces = []
labels = []
label_map = {}
label_id = 0

print("🔁 Training started...")

for person_name in sorted(os.listdir(DATASET_DIR)):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person_name
    face_count = 0

    for img_name in os.listdir(person_path):
        if face_count >= MAX_FACES_PER_LABEL:
            break

        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detected_faces = face_cascade.detectMultiScale(
            img,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in detected_faces:
            if face_count >= MAX_FACES_PER_LABEL:
                break

            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            faces.append(face)
            labels.append(label_id)
            face_count += 1

    print(f"✔ {person_name}: {face_count} faces used")
    label_id += 1

if len(faces) == 0:
    raise RuntimeError("❌ No faces found for training")

# Train model
recognizer.train(faces, np.array(labels))

# Save model
recognizer.write(MODEL_PATH)

# Save label map
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f, indent=4)

print("✅ Training complete")
print(f"📦 Model saved: {MODEL_PATH}")
print(f"🗂️ Labels saved: {LABEL_MAP_PATH}")
print(f"🧠 Total faces trained: {len(faces)}")
