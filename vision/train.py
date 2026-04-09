import face_recognition
import pickle
import os
from PIL import Image

known_encodings = []
known_names = []

for person_name in os.listdir("dataset/"):
    for img_file in os.listdir(f"dataset/{person_name}"):
        img = face_recognition.load_image_file(f"dataset/{person_name}/{img_file}")
        encoding = face_recognition.face_encodings(img)
        known_encodings.append(encoding)
        known_names.append(person_name)

data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)
