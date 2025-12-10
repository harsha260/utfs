# main.py
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FastAPI()

print(">>> LOADING LIGHTWEIGHT AI BRAIN...")
face_engine = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_engine.prepare(ctx_id=0, det_size=(320, 320))
print(">>> AI READY (LOW RAM MODE).")

# In-Memory Database
known_faces_db = []

def get_embedding(img_array):
    faces = face_engine.get(img_array)
    if not faces:
        return None
    # Sort by size (largest face is the user)
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].embedding

def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

@app.get("/")
def home():
    return {"status": "UFPS Brain Online (Low RAM)", "faces_stored": len(known_faces_db)}

@app.post("/scan")
async def scan_face(file: UploadFile = File(...)):
    try:
        # 1. READ IMAGE
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "Invalid Image"}

        # 2. GET EMBEDDING
        embedding = get_embedding(img)
        if embedding is None:
            return {"status": "error", "message": "No face found."}

        # 3. CHECK DUPLICATES
        best_score = 0.0
        for user in known_faces_db:
            score = compute_sim(user['embedding'], embedding)
            if score > best_score:
                best_score = score

        # 4. DECISION
        # 'buffalo_s' is slightly less accurate, so we adjust threshold
        THRESHOLD = 0.50 

        if best_score > THRESHOLD:
            return {
                "status": "duplicate",
                "score": float(best_score),
                "message": "Duplicate Detected!"
            }
        else:
            known_faces_db.append({
                "name": f"User_{len(known_faces_db)+1}",
                "embedding": embedding
            })
            return {
                "status": "approved",
                "score": float(best_score),
                "message": "Access Granted."
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}
