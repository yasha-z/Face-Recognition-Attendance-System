from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
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
import logging

# Initialize FastAPI app
app = FastAPI(
    title="Family Recognition Backend API",
    description="FastAPI backend for React Native family recognition app",
    version="1.0.0"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (same as Flask app)
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
MESSAGE = "WELCOME TO FAMILY RECOGNITION HELPER - Click 'Who is this?' to identify a family member"

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
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    except:
        cap = cv2.VideoCapture(0)

init_camera()

# Create directories (same as Flask)
if not os.path.isdir('FamilyRecords'):
    os.makedirs('FamilyRecords')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'FamilyMembers-{datetoday}.csv' not in os.listdir('FamilyRecords'):
    with open(f'FamilyRecords/FamilyMembers-{datetoday}.csv','w') as f:
        f.write('Name,Relationship,DateAdded')

# Utility functions (exactly from Flask app)
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
            if img is not None:
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.flatten())
                labels.append(user)
    
    if len(faces) > 0:
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        print(f"Model trained with {len(faces)} face samples")

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

# Camera streaming function (same logic as Flask)
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
                cv2.putText(frame, 'Family Recognition Helper', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

# Pydantic models for API responses
class FamilyMemberResponse(BaseModel):
    name: str
    relationship: str
    date_added: str
    id: str

class RecognitionResponse(BaseModel):
    person_name: str
    confidence: Optional[float] = None
    status: str

# Custom exception handler for better debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for {request.method} {request.url}: {exc.errors()}")
    logger.error(f"Request body type: {type(exc.body)}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Validation error - check your request format",
            "received_body_type": str(type(exc.body))
        }
    )

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Family Recognition Backend API",
        "version": "1.0.0",
        "total_family_members": totalreg(),
        "date": datetoday2,
        "status": "ready"
    }

@app.get("/api/status")
async def get_status():
    """Get overall system status"""
    return {
        "camera_available": cap is not None and cap.isOpened(),
        "total_family_members": totalreg(),
        "model_trained": os.path.exists('static/face_recognition_model.pkl'),
        "date": datetoday2,
        "message": MESSAGE
    }

@app.get("/api/family-members")
async def get_family_members():
    """Get list of all family members"""
    names, relationships, dates, count = extract_family_members()
    
    family_members = []
    for i in range(count):
        family_members.append({
            "name": names.iloc[i],
            "relationship": relationships.iloc[i], 
            "date_added": dates.iloc[i],
            "id": f"{names.iloc[i]}_{relationships.iloc[i]}"
        })
    
    return {
        "family_members": family_members,
        "total_count": count
    }

# Make the form fields optional to handle React Native requests better
@app.post("/api/family-members")
async def add_family_member(
    background_tasks: BackgroundTasks,
    name: Optional[str] = Form(None),
    relationship: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None)
):
    """Add a new family member with their photos"""
    
    logger.info(f"Received add_family_member request: name={name}, relationship={relationship}")
    
    # Validate required fields
    if not name or not relationship:
        raise HTTPException(status_code=400, detail="Name and relationship are required")
    
    # Create unique identifier
    member_id = random.randint(1000, 9999)
    userimagefolder = f'static/faces/{name}_{relationship}_{member_id}'
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    # Save uploaded images
    saved_count = 0
    if images and len(images) > 0:
        for i, image_file in enumerate(images):
            if image_file and hasattr(image_file, 'content_type') and image_file.content_type:
                if image_file.content_type.startswith('image/'):
                    try:
                        contents = await image_file.read()
                        
                        # Convert to OpenCV format
                        nparr = np.frombuffer(contents, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            # Save the image
                            filename = f'{name}_{i}.jpg'
                            cv2.imwrite(f'{userimagefolder}/{filename}', img)
                            saved_count += 1
                    except Exception as e:
                        logger.error(f"Error processing image {i}: {e}")
                        continue
    
    # Record family member in CSV
    with open(f'FamilyRecords/FamilyMembers-{datetoday}.csv', 'a') as f:
        f.write(f'\n{name},{relationship},{datetoday2}')
    
    # Train model in background if we have images
    if saved_count > 0:
        background_tasks.add_task(train_model)
    
    return {
        "message": f"Successfully added {name} ({relationship})",
        "images_saved": saved_count,
        "total_family_members": totalreg(),
        "status": "success"
    }

@app.get("/api/camera/feed")
async def video_feed():
    """Get camera video stream for web viewing"""
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

@app.post("/api/recognition/start")
async def start_recognition():
    """Start face recognition"""
    global recognition_active, recognized_person, camera_active
    
    camera_active = True
    recognition_active = True
    recognized_person = None
    
    # Give time for recognition to process
    await asyncio.sleep(3)
    
    return {
        "person_name": recognized_person if recognized_person else 'No face detected',
        "status": 'active',
        "recognition_active": recognition_active
    }

@app.post("/api/recognition/stop")
async def stop_recognition():
    """Stop face recognition"""
    global recognition_active
    recognition_active = False
    return {"status": "recognition_stopped", "active": recognition_active}

@app.get("/api/recognition/status")
async def get_recognition_status():
    """Get current recognition status"""
    return {
        "person_name": recognized_person if recognized_person else 'No recognition active',
        "status": 'active' if recognition_active else 'inactive',
        "recognition_active": recognition_active
    }

@app.post("/api/recognition/identify")
async def identify_person_from_image(image: UploadFile = File(...)):
    """Identify person from uploaded image - main endpoint for React Native"""
    
    logger.info(f"Received identify request: filename={image.filename}, content_type={image.content_type}")
    
    try:
        if not image:
            raise HTTPException(status_code=400, detail="No image provided")
            
        # Read and process image
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image file")
            
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
    # Detect faces
    faces = extract_faces(img)
    
    if len(faces) == 0:
        return {
            "person_name": "No face detected",
            "status": "no_face",
            "confidence": None
        }
    
    # Use the first detected face
    x, y, w, h = faces[0]
    face_roi = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (50, 50))
    
    try:
        if os.path.exists('static/face_recognition_model.pkl'):
            identified_person = identify_face(face_resized.reshape(1, -1))[0]
            person_parts = identified_person.split('_')
            display_name = person_parts[0] if person_parts else identified_person
            
            return {
                "person_name": display_name,
                "status": "identified",
                "confidence": 0.85  # Mock confidence for now
            }
        else:
            return {
                "person_name": "No trained model available",
                "status": "no_model",
                "confidence": None
            }
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return {
            "person_name": "Recognition failed",
            "status": "error",
            "confidence": None
        }

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

# Debug endpoint for testing
@app.post("/api/debug/upload-test")
async def debug_upload_test(
    name: Optional[str] = Form(None),
    relationship: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Debug endpoint to test file upload format"""
    result = {
        "name": name,
        "relationship": relationship,
        "image_info": None
    }
    
    if image:
        contents = await image.read()
        result["image_info"] = {
            "filename": image.filename,
            "content_type": image.content_type,
            "size": len(contents)
        }
    
    return result

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)