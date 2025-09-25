import sqlite3
import cv2
import os
from flask import Flask,request,render_template,redirect,session,url_for,Response,jsonify
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time
import threading
# import db

#VARIABLES
MESSAGE = "WELCOME TO FAMILY RECOGNITION HELPER" \
          " Click 'Who is this?' to identify a family member"

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

# Global variables for camera streaming
camera_active = False
recognition_active = False
current_frame = None
recognized_person = None

#### If these directories don't exist, create them
if not os.path.isdir('FamilyRecords'):
    os.makedirs('FamilyRecords')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'FamilyMembers-{datetoday}.csv' not in os.listdir('FamilyRecords'):
    with open(f'FamilyRecords/FamilyMembers-{datetoday}.csv','w') as f:
        f.write('Name,Relationship,DateAdded')

#### get a number of total registered users

def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    if img is not None and img.size > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

#### get a list of family members
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

#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')
    else:
        print("this user has already marked attendence for the day , but still i am marking it ")
        # with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
        #     f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS ##############################

#### Camera streaming functions
def generate_frames():
    global current_frame, recognition_active, recognized_person, cap
    
    # Initialize camera if not already done
    if cap is None or not cap.isOpened():
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = cv2.VideoCapture(1)
        except:
            cap = cv2.VideoCapture(0)
    
    while camera_active:
        success, frame = cap.read()
        if not success:
            # Create a black frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'Camera not available', (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            current_frame = frame.copy()
            
            if recognition_active:
                # Detect faces
                faces = extract_faces(frame)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Try to identify the person
                    face_roi = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (50, 50))
                    
                    try:
                        if os.path.exists('static/face_recognition_model.pkl'):
                            identified_person = identify_face(face_resized.reshape(1, -1))[0]
                            # Extract name from the identifier
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
                        print(f"Recognition error: {e}")
            
            # Add instruction text
            if recognition_active:
                cv2.putText(frame, 'Looking for faces...', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(frame, 'Camera ready - Click Start to recognize', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

@app.route('/video_feed')
def video_feed():
    global camera_active
    camera_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition')
def start_recognition():
    global recognition_active, recognized_person, camera_active
    camera_active = True
    recognition_active = True
    recognized_person = None
    
    # Give some time for recognition to process
    import time
    time.sleep(3)
    
    return jsonify({
        'person_name': recognized_person if recognized_person else 'No face detected',
        'status': 'active'
    })

@app.route('/stop_recognition')
def stop_recognition():
    global recognition_active, camera_active
    recognition_active = False
    camera_active = False
    return jsonify({'status': 'stopped'})

@app.route('/get_recognition_status')
def get_recognition_status():
    global recognized_person, recognition_active
    return jsonify({
        'person_name': recognized_person,
        'active': recognition_active
    })

#### Our main page
@app.route('/')
def home():
    return render_template('home_new.html', totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)


#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    ATTENDENCE_MARKED = False
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us , kindly register yourself first'
        print("face not in database, need to register")
        return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg,datetoday2=datetoday2, mess = MESSAGE)
        # return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                current_time_ = datetime.now().strftime("%H:%M:%S")
                print(f"attendence marked for {identified_person}, at {current_time_} ")
                ATTENDENCE_MARKED = True
                break
        if ATTENDENCE_MARKED:
            # time.sleep(3)
            break

        # Display the resulting frame
        cv2.imshow('Attendance Check, press "q" to exit', frame)
        cv2.putText(frame,'hello',(30,30),cv2.FONT_HERSHEY_COMPLEX,2,(255, 255, 255))
        
    # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendence taken successfully'
    print("attendence registered")
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, mess=MESSAGE)

@app.route('/add_family',methods=['GET','POST'])
def add_family():
    familyMemberName = request.form['familyMemberName']
    relationship = request.form['relationship']
    
    # Create unique identifier
    import random
    member_id = random.randint(1000, 9999)
    userimagefolder = f'static/faces/{familyMemberName}_{relationship}_{member_id}'
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    # Use a separate camera instance for capturing
    capture_cap = cv2.VideoCapture(0)
    i, j = 0, 0
    
    while i < 50:  # Capture exactly 50 images
        ret, frame = capture_cap.read()
        if not ret:
            continue
            
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Capturing photos: {i}/50', (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Adding: {familyMemberName}', (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Look at the camera. Press ESC to stop', (30, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            if j % 10 == 0:  # Capture every 10th frame
                name = f'{familyMemberName}_{i}.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
            
        cv2.imshow(f'Adding Family Member: {familyMemberName}', frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break
    
    capture_cap.release()
    cv2.destroyAllWindows()
    
    # Record family member in CSV
    with open(f'FamilyRecords/FamilyMembers-{datetoday}.csv', 'a') as f:
        f.write(f'\n{familyMemberName},{relationship},{datetoday2}')
    
    print('Training Model with new family member...')
    train_model()
    
    MESSAGE = f'Successfully added {familyMemberName} ({relationship}) to your family recognition system!'
    return render_template('home_new.html', totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)
    # return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)

#### Our main function which runs the Flask App
app.run(debug=True,port=1000)
if __name__ == '__main__':
    pass
#### This function will run when we add a new user
