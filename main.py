# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

app = FastAPI()

# --- 1. SETUP AI ENGINE (Run once on startup) ---
print(">>> LOADING AI BRAIN... (This takes 10-20 seconds)")
# We use 'buffalo_l' (Standard) or 'buffalo_s' (Faster/Lighter)
# ctx_id=0 means use GPU if available, else CPU (-1)
face_engine = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_engine.prepare(ctx_id=0, det_size=(640, 640))
print(">>> AI READY.")

# --- 2. TEMPORARY DATABASE ---
# In a real app, you would use PostgreSQL.
# Here, we store faces in memory (RAM). If you restart the server, this wipes.
# Format: [{'name': 'Visitor_1', 'embedding': [vector...]}]
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
    return {"status": "UFPS Brain Online", "faces_stored": len(known_faces_db)}

@app.post("/scan")
async def scan_face(file: UploadFile = File(...)):
    # 1. READ IMAGE FROM PHONE
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "Invalid Image"}

    # 2. EXTRACT FACE DATA
    embedding = get_embedding(img)
    if embedding is None:
        return {"status": "error", "message": "No face visible. Remove masks/glasses."}

    # 3. CHECK DATABASE FOR DUPLICATES
    best_score = 0.0
    
    for user in known_faces_db:
        score = compute_sim(user['embedding'], embedding)
        if score > best_score:
            best_score = score

    # 4. DECISION THRESHOLD
    THRESHOLD = 0.50  # Strictness (0.4 to 0.6 is normal)

    if best_score > THRESHOLD:
        return {
            "status": "duplicate",
            "score": float(best_score),
            "message": "Double Dipping Detected!"
        }
    else:
        # REGISTER NEW USER
        known_faces_db.append({
            "name": f"User_{len(known_faces_db)+1}",
            "embedding": embedding
        })
        return {
            "status": "approved",
            "score": float(best_score),
            "message": "New User Registered."
        }