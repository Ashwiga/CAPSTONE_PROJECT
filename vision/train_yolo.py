import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet

# Load YOLO face detector
yolo_model = YOLO("models/yolov8n-face.pt")# or your model path

# Load FaceNet embedder
embedder = FaceNet()

dataset_path = "dataset"   # dataset/person_name/image.jpg
save_path = "embeddings/faces.pkl"

faces_db = {}

print("🚀 Training started...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"\n📌 Processing person: {person_name}")

    faces_db[person_name] = []

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)

        img = cv2.imread(img_path)

        if img is None:
            print("❌ Unable to read:", img_path)
            continue

        # YOLO detect face
        results = yolo_model(img)

        if len(results[0].boxes) == 0:
            print(f"⚠ No face detected in {img_file}")
            continue

        # Take best face (highest confidence)
        best_box = None
        best_conf = 0

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

        x1, y1, x2, y2 = best_box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Add padding (important!)
        pad = 25
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad)
        y2 = min(img.shape[0], y2 + pad)

        face_crop = img[y1:y2, x1:x2]

        if face_crop.size == 0:
            print("⚠ Invalid crop:", img_file)
            continue

        # Resize to FaceNet input
        face_crop = cv2.resize(face_crop, (160, 160))

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Get embedding
        embedding = embedder.embeddings([face_rgb])[0]

        # Save embedding
        faces_db[person_name].append(embedding)

        print(f"✅ {img_file} embedding saved (conf={best_conf:.2f})")

# Save faces.pkl
os.makedirs("embeddings", exist_ok=True)

with open(save_path, "wb") as f:
    pickle.dump(faces_db, f)

print("\n✅ Training complete!")
print("📌 Embeddings saved in:", save_path)

# Print summary
for name in faces_db:
    print(f"{name} -> {len(faces_db[name])} embeddings")
