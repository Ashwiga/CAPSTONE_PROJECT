import cv2
import os
import json
import numpy as np

# ---------------- PATHS ----------------
DATASET_DIR = "dataset"
OUTPUT_DIR = "embeddings"
MODEL_PATH = os.path.join(OUTPUT_DIR, "faces.yml")
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, "label_map.json")

DNN_PROTO = "models/deploy.prototxt"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

MAX_FACES_PER_PERSON = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD DNN ----------------
net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)

# ---------------- LBPH ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
label_id = 0

print("🔁 Training started...")

# ---------------- TRAIN LOOP ----------------
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person_name
    face_count = 0

    for img_name in os.listdir(person_path):
        if face_count >= MAX_FACES_PER_PERSON:
            break

        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                face = image[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (200, 200))

                faces.append(gray)
                labels.append(label_id)
                face_count += 1

                if face_count >= MAX_FACES_PER_PERSON:
                    break

    print(f"✔ {person_name}: {face_count} faces")
    label_id += 1

# ---------------- TRAIN MODEL ----------------
if len(faces) == 0:
    raise RuntimeError("❌ No faces found for training")

recognizer.train(faces, np.array(labels))

# ---------------- SAVE ----------------
recognizer.write(MODEL_PATH)

with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f, indent=4)

print("✅ Training complete")
print(f"📦 Model saved: {MODEL_PATH}")
print(f"🗂️ Labels saved: {LABEL_MAP_PATH}")