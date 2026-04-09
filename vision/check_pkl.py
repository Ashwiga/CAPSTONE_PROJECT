
import pickle

with open("embeddings/faces.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
print("Total persons:", len(data))
