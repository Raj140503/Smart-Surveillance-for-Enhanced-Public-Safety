import os
import json
import subprocess
import shutil
import cv2
import tempfile
import time
import threading

from flask import send_from_directory   
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from werkzeug.utils import secure_filename
from settings.settings import CAMERA, FACE_DETECTION, PATHS

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('images', exist_ok=True)
os.makedirs('templates', exist_ok=True)  # Ensuring templates directory exists

# Global variables
DETECTION_ACTIVE = False
CAMERA_FEED = None
face_cascade = None
recognizer = None
names = {}


# Helper functions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_names_data():
    """Load names from the JSON file"""
    if os.path.exists('names.json'):
        with open('names.json', 'r') as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}

def save_names_data(names_data):
    """Save names to the JSON file"""
    with open('names.json', 'w') as f:
        json.dump(names_data, f, indent=4)




@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)


def get_people_with_images():
    names_data = get_names_data()
    people = []

    for id_str, name in names_data.items():
        all_images = [
            filename for filename in os.listdir('images')
            if filename.startswith(f'Users-{id_str}-')
        ]
        if all_images:
            people.append({
                'id': id_str,
                'name': name,
                'first_image': url_for('serve_image', filename=all_images[0]),
                'all_images': [url_for('serve_image', filename=f) for f in all_images]
            })

    return people




def get_next_id():
    """Get the next available ID"""
    names_data = get_names_data()
    return max(map(int, names_data.keys()), default=0) + 1

def initialize_detection_system():
    """Initialize face recognition components"""
    global face_cascade, recognizer, names
    
    try:
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists(PATHS['trainer_file']):
            recognizer.read(PATHS['trainer_file'])
        else:
            print("Warning: Trainer file not found")
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            print("Error loading cascade classifier")
            return False
        
        # Load names
        names = get_names_data()
        
        return True
    except Exception as e:
        print(f"Error initializing detection system: {e}")
        return False

def initialize_camera():
    """Initialize the camera"""
    global CAMERA_FEED
    try:
        CAMERA_FEED = cv2.VideoCapture(CAMERA['index'])
        if not CAMERA_FEED.isOpened():
            print("Could not open webcam")
            return False
        
        CAMERA_FEED.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        CAMERA_FEED.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def release_camera():
    """Release the camera"""
    global CAMERA_FEED
    if CAMERA_FEED is not None:
        CAMERA_FEED.release()
        CAMERA_FEED = None

def generate_frames():
    """Generate video frames for streaming"""
    global DETECTION_ACTIVE, CAMERA_FEED, face_cascade, recognizer, names
    
    if not initialize_camera():
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + open('static/no_camera.jpg', 'rb').read() + b'\r\n'
        return
    
    if not initialize_detection_system():
        while CAMERA_FEED and CAMERA_FEED.isOpened():
            success, frame = CAMERA_FEED.read()
            if not success:
                break
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            if not DETECTION_ACTIVE:
                break
        
        release_camera()
        return
    
    while CAMERA_FEED and CAMERA_FEED.isOpened() and DETECTION_ACTIVE:
        success, img = CAMERA_FEED.read()
        if not success:
            break
        
        if DETECTION_ACTIVE:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            for (x, y, w, h) in faces:
                # Recognize the face
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Check confidence and determine name
                if confidence <= 100:
                    name = names.get(str(id), "unauthorized")
                    confidence_text = f"{confidence:.1f}%"
                    
                    # Set bounding box color based on authorization status
                    color = (0, 255, 0) if name != "unauthorized" else (0, 0, 255)
                    
                    # Draw rectangle
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    
                    # Display name and confidence
                    cv2.putText(img, name, (x+5, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(img, confidence_text, (x+5, y+h-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        
        # Encode the frame to jpg
        ret, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    release_camera()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/database')
def database():
    people = get_people_with_images()
    return render_template('database.html', people=people)

@app.route('/detection')
def detection():
    if not os.path.exists('templates/detection.html'):
        return "Error: detection.html file is missing!", 500
    return render_template('detection.html')

@app.route('/api/people', methods=['GET'])
def get_people():
    return jsonify(get_people_with_images())

@app.route('/api/people/<id>', methods=['DELETE'])
def delete_person(id):
    names_data = get_names_data()
    
    if id in names_data:
        del names_data[id]
        save_names_data(names_data)
    
        for filename in os.listdir('images'):
            if filename.startswith(f'Users-{id}-'):
                os.remove(os.path.join('images', filename))
    
        try:
            subprocess.run(['python', 'face_trainer.py'], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': True})

@app.route('/api/add_person', methods=['POST'])
def add_person():
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400
    
    temp_name_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_name_path = temp_file.name
            temp_file.write(name)
        
        subprocess.run(['python', 'face_taker.py'], input=name, text=True, check=True)
        subprocess.run(['python', 'face_trainer.py'], check=True)
        
        return jsonify({'success': True})
    except subprocess.CalledProcessError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if temp_name_path and os.path.exists(temp_name_path):
            os.remove(temp_name_path)

@app.route('/api/start_live_detection', methods=['POST'])
def start_live_detection():
    """Starts the face detection in the web UI"""
    global DETECTION_ACTIVE
    
    if not DETECTION_ACTIVE:
        DETECTION_ACTIVE = True
        return jsonify({"success": True, "message": "Detection started"})
    else:
        return jsonify({"success": False, "error": "Detection is already running"})

@app.route('/api/stop_live_detection', methods=['POST'])
def stop_live_detection():
    """Stops the face detection in the web UI"""
    global DETECTION_ACTIVE
    
    if DETECTION_ACTIVE:
        DETECTION_ACTIVE = False
        return jsonify({"success": True, "message": "Detection stopped"})
    else:
        return jsonify({"success": False, "error": "No active detection process"})

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    name = request.form.get('name', '').strip()
    file = request.files.get('file')
    
    if not name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    if not allowed_file(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return jsonify({'success': False, 'error': 'Invalid file format'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    face_id = get_next_id()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        return jsonify({'success': False, 'error': 'Error loading cascade classifier'}), 500
    
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    if not faces:
        return jsonify({'success': False, 'error': 'No face detected'}), 400
    
    for i, (x, y, w, h) in enumerate(faces[:20]):
        cv2.imwrite(f'./images/Users-{face_id}-{i+1}.jpg', gray[y:y+h, x:x+w])
    
    names_data = get_names_data()
    names_data[str(face_id)] = name
    save_names_data(names_data)
    
    try:
        subprocess.run(['python', 'face_trainer.py'], check=True)
        return jsonify({'success': True})
    except subprocess.CalledProcessError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        os.remove(filepath)

@app.route('/video_feed')
def video_feed():
    """Route to stream live video feed"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)