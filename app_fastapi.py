from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time
import threading
from datetime import date, datetime
import random
import base64
import io
import asyncio
from PIL import Image

# Initialize FastAPI app
app = FastAPI(
    title="Family Recognition API",
    description="Backend API for dementia patient family recognition system",
    version="1.0.0"
)

# Add CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Camera and recognition globals
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera_active = False
recognition_active = False
current_frame = None
recognized_person = None
cap = None

# Initialize camera
def init_camera():
    global cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
    except:
        cap = cv2.VideoCapture(0)

init_camera()

# Create directories
if not os.path.isdir('FamilyRecords'):
    os.makedirs('FamilyRecords')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'FamilyMembers-{datetoday}.csv' not in os.listdir('FamilyRecords'):
    with open(f'FamilyRecords/FamilyMembers-{datetoday}.csv','w') as f:
        f.write('Name,Relationship,DateAdded')

# Pydantic models for API
class FamilyMemberRequest(BaseModel):
    name: str
    relationship: str

class FamilyMemberResponse(BaseModel):
    name: str
    relationship: str
    date_added: str
    id: str

class RecognitionResponse(BaseModel):
    person_name: str
    confidence: Optional[float] = None
    status: str

class CameraStatusResponse(BaseModel):
    active: bool
    recognition_active: bool

# Utility functions
def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if img is not None and img.size > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.flatten())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_family_members():
    try:
        df = pd.read_csv(f'FamilyRecords/FamilyMembers-{datetoday}.csv')
        names = df['Name']
        relationships = df['Relationship']
        dates = df['DateAdded']
        l = len(df)
        return names, relationships, dates, l
    except:
        return [], [], [], 0

# Camera streaming function
def generate_frames():
    global current_frame, recognition_active, recognized_person, cap
    
    if cap is None or not cap.isOpened():
        init_camera()
    
    while camera_active:
        success, frame = cap.read()
        if not success:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'Camera not available', (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            current_frame = frame.copy()
            
            if recognition_active:
                faces = extract_faces(frame)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    face_roi = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (50, 50))
                    
                    try:
                        if os.path.exists('static/face_recognition_model.pkl'):
                            identified_person = identify_face(face_resized.reshape(1, -1))[0]
                            person_parts = identified_person.split('_')
                            display_name = person_parts[0] if person_parts else identified_person
                            
                            cv2.putText(frame, f'{display_name}', (x + 6, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            recognized_person = display_name
                        else:
                            cv2.putText(frame, 'No trained model', (x + 6, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            recognized_person = 'No trained faces'
                    except Exception as e:
                        cv2.putText(frame, 'Unknown', (x + 6, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        recognized_person = 'Unknown'
            
            if recognition_active:
                cv2.putText(frame, 'Looking for faces...', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(frame, 'Camera ready', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Family Recognition API",
        "version": "1.0.0",
        "total_family_members": totalreg(),
        "date": datetoday2
    }

@app.get("/api/status")
async def get_status():
    """Get overall system status"""
    return {
        "camera_available": cap is not None and cap.isOpened(),
        "total_family_members": totalreg(),
        "model_trained": os.path.exists('static/face_recognition_model.pkl'),
        "date": datetoday2
    }

@app.get("/api/family-members", response_model=List[FamilyMemberResponse])
async def get_family_members():
    """Get list of all family members"""
    names, relationships, dates, count = extract_family_members()
    
    family_members = []
    for i in range(count):
        family_members.append(FamilyMemberResponse(
            name=names.iloc[i],
            relationship=relationships.iloc[i],
            date_added=dates.iloc[i],
            id=f"{names.iloc[i]}_{relationships.iloc[i]}"
        ))
    
    return family_members

@app.post("/api/family-members", response_model=dict)
async def add_family_member(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    relationship: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """Add a new family member with their photos"""
    
    # Create unique identifier
    member_id = random.randint(1000, 9999)
    userimagefolder = f'static/faces/{name}_{relationship}_{member_id}'
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    # Save uploaded images
    saved_count = 0
    for i, image_file in enumerate(images):
        if image_file.content_type.startswith('image/'):
            contents = await image_file.read()
            
            # Convert to OpenCV format
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Save the image
                filename = f'{name}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{filename}', img)
                saved_count += 1
    
    # Record family member in CSV
    with open(f'FamilyRecords/FamilyMembers-{datetoday}.csv', 'a') as f:
        f.write(f'\n{name},{relationship},{datetoday2}')
    
    # Train model in background
    background_tasks.add_task(train_model)
    
    return {
        "message": f"Successfully added {name} ({relationship})",
        "images_saved": saved_count,
        "total_family_members": totalreg()
    }

@app.get("/api/camera/feed")
async def video_feed():
    """Get camera video stream"""
    global camera_active
    camera_active = True
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/camera/start")
async def start_camera():
    """Start camera feed"""
    global camera_active
    camera_active = True
    init_camera()
    return {"status": "camera_started", "active": camera_active}

@app.post("/api/camera/stop")
async def stop_camera():
    """Stop camera feed"""
    global camera_active, recognition_active
    camera_active = False
    recognition_active = False
    return {"status": "camera_stopped", "active": camera_active}

@app.get("/api/camera/status", response_model=CameraStatusResponse)
async def get_camera_status():
    """Get camera status"""
    return CameraStatusResponse(
        active=camera_active,
        recognition_active=recognition_active
    )

@app.post("/api/recognition/start", response_model=RecognitionResponse)
async def start_recognition():
    """Start face recognition"""
    global recognition_active, recognized_person, camera_active
    
    camera_active = True
    recognition_active = True
    recognized_person = None
    
    # Give time for recognition to process
    await asyncio.sleep(3)
    
    return RecognitionResponse(
        person_name=recognized_person if recognized_person else 'No face detected',
        status='active'
    )

@app.post("/api/recognition/stop")
async def stop_recognition():
    """Stop face recognition"""
    global recognition_active
    recognition_active = False
    return {"status": "recognition_stopped", "active": recognition_active}

@app.get("/api/recognition/status", response_model=RecognitionResponse)
async def get_recognition_status():
    """Get current recognition status"""
    return RecognitionResponse(
        person_name=recognized_person if recognized_person else 'No recognition active',
        status='active' if recognition_active else 'inactive'
    )

@app.post("/api/recognition/identify")
async def identify_person_from_image(image: UploadFile = File(...)):
    """Identify person from uploaded image"""
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and process image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Detect faces
    faces = extract_faces(img)
    
    if len(faces) == 0:
        return RecognitionResponse(
            person_name="No face detected",
            status="no_face"
        )
    
    # Use the first detected face
    x, y, w, h = faces[0]
    face_roi = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (50, 50))
    
    try:
        if os.path.exists('static/face_recognition_model.pkl'):
            identified_person = identify_face(face_resized.reshape(1, -1))[0]
            person_parts = identified_person.split('_')
            display_name = person_parts[0] if person_parts else identified_person
            
            return RecognitionResponse(
                person_name=display_name,
                status="identified"
            )
        else:
            return RecognitionResponse(
                person_name="No trained model available",
                status="no_model"
            )
    except Exception as e:
        return RecognitionResponse(
            person_name="Recognition failed",
            status="error"
        )

@app.delete("/api/family-members/{member_id}")
async def delete_family_member(member_id: str, background_tasks: BackgroundTasks):
    """Delete a family member and retrain model"""
    
    # Find and delete the member's folder
    faces_dir = 'static/faces'
    deleted = False
    
    for folder in os.listdir(faces_dir):
        if member_id in folder:
            import shutil
            shutil.rmtree(os.path.join(faces_dir, folder))
            deleted = True
            break
    
    if deleted:
        # Retrain model in background
        background_tasks.add_task(train_model)
        return {"message": f"Family member {member_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Family member not found")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)