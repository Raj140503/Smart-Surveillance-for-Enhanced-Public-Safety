from flask import Flask, Response, render_template, jsonify, send_from_directory
import cv2
import csv
import os
import json
import threading
import time
from datetime import datetime
import logging
from settings.settings import CAMERA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the face recognizer functionality
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_recognizer import (
    initialize_camera, 
    load_names, 
    load_email_settings,
    cv2,
    PATHS, 
    FACE_DETECTION,
    RECOGNITION
)

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
face_cascade = None
recognizer = None
names = {}
running = False  # Start with detection off
detection_thread = None
import numpy as np  # Import numpy at the module level

def initialize_detection():
    """Initialize all components needed for face detection"""
    global camera, face_cascade, recognizer, names, running
    
    try:
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(PATHS['trainer_file']):
            logger.error("Trainer file not found. Please train the model first.")
            return False
        recognizer.read(PATHS['trainer_file'])
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            logger.error("Error loading cascade classifier")
            return False
        
        # Initialize camera
        camera = initialize_camera(CAMERA['index'])
        if camera is None:
            logger.error("Failed to initialize camera")
            return False
        
        # Load names
        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("No names loaded, recognition will be limited")
        
        running = True
        return True
    except Exception as e:
        logger.error(f"Error initializing detection: {e}")
        return False

def detect_faces():
    """Process frames from camera and perform face detection/recognition"""
    global output_frame, camera, lock, running, face_cascade, recognizer, names
    
    log_file = "detection_log.csv"
    intruder_detected_time = None
    last_email_time = 0
    last_log_time = 0
    cooldown_time = 20  # seconds
    detection_time = 5   # seconds
    log_interval = 10  # seconds
    email_settings = load_email_settings()
    
    # Ensure log file exists with headers
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Name', 'Confidence', 'Status', 'Image_Path'])

    # Ensure intruders directory exists
    os.makedirs("intruders", exist_ok=True)
    
    while running:
        try:
            success, img = camera.read()
            if not success:
                logger.warning("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            intruder_present = False
            for (x, y, w, h) in faces:
                # Recognize the face
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                face_image = img[y:y+h, x:x+w]
                image_path = ""

                THRESHOLD = RECOGNITION['confidence_threshold']

                # Check confidence and determine name
                if confidence > THRESHOLD:
                    name = "unauthorized"
                else:
                    name = names.get(str(id), "unauthorized")

                confidence_text = f"{confidence:.1f}%"
                color = (0, 255, 0) if name != "unauthorized" else (0, 0, 255)

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, name, (x+5, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, confidence_text, (x+5, y+h-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                
                # Add timestamp to the frame
                cv2.putText(img, timestamp, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if name == "unauthorized":
                    status = "unauthorized"
                    intruder_present = True

                    # Save cropped image
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_name = f"intruder_{ts}.jpg"

                    image_path = os.path.join("intruders", image_name)
                    cv2.imwrite(image_path, face_image)

                    full_frame_name = f"frame_{int(time.time())}.jpg"
                    full_frame_path = os.path.join("intruders", full_frame_name)
                    cv2.imwrite(full_frame_path, img)  # Save full frame as well
                    
                    if intruder_detected_time is None:
                        intruder_detected_time = time.time()
                    
                    elapsed_time = time.time() - intruder_detected_time
                    if elapsed_time >= detection_time:
                        if time.time() - last_email_time >= cooldown_time:
                            # Import here to avoid circular import
                            from email_alert import send_alert
                            threading.Thread(target=send_alert, 
                                           args=(email_settings, full_frame_path, image_path)).start()
                            last_email_time = time.time()
                            intruder_detected_time = None
                else:
                    status = "authorized"
                    image_path = ""

                # Log the detection
                if time.time() - last_log_time >= log_interval:
                    try:
                        with open(log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, name, f"{confidence:.2f}", status, image_path])
                        last_log_time = time.time()
                    except Exception as e:
                        logger.error(f"Failed to write log entry: {e}")
                        
            if not intruder_present:
                intruder_detected_time = None  # Reset detection timer if no intruder
                
            # Update the output frame with lock protection
            with lock:
                output_frame = img.copy()
                
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
    
    # Release resources when done
    if camera is not None:
        camera.release()
        logger.info("Camera released")

def generate_frames():
    """Generate frames for the video stream"""
    global output_frame, lock, running
    
    while True:
        # Wait until we have a frame
        if output_frame is None or not running:
            # If not running, generate a placeholder frame
            placeholder = create_placeholder_frame()
            (flag, encoded_image) = cv2.imencode(".jpg", placeholder)
            if flag:
                yield (b'--frame\r\n' 
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       bytearray(encoded_image) + b'\r\n')
            time.sleep(0.1)
            continue
            
        # Encode the frame in JPEG format
        with lock:
            if output_frame is None:
                continue
                
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
                
        # Yield the frame in the byte format
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encoded_image) + b'\r\n')

def create_placeholder_frame():
    """Create a placeholder frame when detection is not running"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Detection Stopped", (160, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Click 'Start Detection' to begin", (120, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

@app.route('/')
def index():
    """Redirect to detection page"""
    return render_template('index.html')

@app.route('/detection')
def detection():
    """Render detection page"""
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    """Stream video feed to client"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    """Return detection logs as JSON"""
    log_data = []
    try:
        log_file = "detection_log.csv"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    log_data.append(row)
        return jsonify({"logs": log_data})
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/intruders/<path:filename>')
def serve_intruder_image(filename):
    """Serve intruder images"""
    return send_from_directory('intruders', filename)

@app.route('/api/status')
def system_status():
    """Return system status"""
    global running
    return jsonify({
        "running": running,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/start_detection', methods=['POST'])
def api_start_detection():
    """API endpoint to start detection"""
    global running, detection_thread
    
    if running:
        return jsonify({
            "success": False,
            "message": "Detection system is already running"
        })
        
    if start_detection():
        return jsonify({
            "success": True,
            "message": "Detection system started successfully"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to start detection system"
        })

@app.route('/api/stop_detection', methods=['POST'])
def api_stop_detection():
    """API endpoint to stop detection"""
    global running, camera
    
    if not running:
        return jsonify({
            "success": False,
            "message": "Detection system is not running"
        })
    
    # Stop the detection loop
    running = False
    
    # Wait for the thread to finish
    time.sleep(1)
    
    return jsonify({
        "success": True,
        "message": "Detection system stopped successfully"
    })

def start_detection():
    """Start the detection thread"""
    global running, detection_thread
    
    # If already running, don't start again
    if running:
        return True
        
    if initialize_detection():
        detection_thread = threading.Thread(target=detect_faces)
        detection_thread.daemon = True
        detection_thread.start()
        logger.info("Detection system started")
        return True
    return False

if __name__ == '__main__':
    # Start with detection system stopped initially
    running = False
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5004, debug=False)