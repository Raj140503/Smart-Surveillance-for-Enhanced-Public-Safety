#face_recognizer.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import csv
from datetime import datetime
import cv2
import numpy as np
import json
import os
import logging
import time
import threading
from settings.settings import CAMERA, FACE_DETECTION, PATHS
from settings.settings import RECOGNITION
from email_alert import send_alert, capture_intruder, load_email_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Initialize the camera with error handling
    """
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None
        
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

def load_names(filename: str) -> dict:
    """
    Load name mappings from JSON file
    """
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}

def send_intruder_alert(email_settings, full_image, intruder_image):
    """
    Send an email alert with two images: full screen and intruder close-up.
    """
    full_image_path = "full_screen.jpg"
    intruder_image_path = "intruder.jpg"
    
    cv2.imwrite(full_image_path, full_image)
    cv2.imwrite(intruder_image_path, intruder_image)
    
    send_alert(email_settings, full_image_path, intruder_image_path)


if __name__ == "__main__":
    try:
        logger.info("Starting Intrusion Detection system...")
        
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")
        recognizer.read(PATHS['trainer_file'])
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")
        
        # Initialize camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")
        
        # Load names
        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("No names loaded, recognition will be limited")
        
        # Load email settings
        email_settings = load_email_settings()
        alert_enabled = email_settings.get("ALERT", False)
        
        logger.info("Press 'CTRL + C' to exit.")
        
        intruder_detected_time = None
        last_email_time = 0
        cooldown_time = 20  # seconds
        detection_time = 5   # seconds
        # Add this near the top, just after other timers
        log_interval = 10  # seconds
        last_log_time = 0

        
        log_file = "detection_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Name', 'Confidence', 'Status', 'Image_Path'])

        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
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

                    
                    if name == "unauthorized":
                        status = "unauthorized"
                        intruder_present = True

                        # Save cropped image
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_name = f"intruder_{ts}.jpg"

                        crop_image_path = os.path.join("intruders", image_name)
                        os.makedirs("intruders", exist_ok=True)
                        cv2.imwrite(image_path, face_image)

                        full_frame_name = f"frame_{int(time.time())}.jpg"
                        image_path = os.path.join("intruders", full_frame_name)
                        cv2.imwrite(full_frame_path, img)  # Save full frame as well
                        
                        if intruder_detected_time is None:
                            intruder_detected_time = time.time()
                        
                        elapsed_time = time.time() - intruder_detected_time
                        if elapsed_time >= detection_time:
                            if time.time() - last_email_time >= cooldown_time:
                                threading.Thread(target=send_intruder_alert, args=(email_settings, img, face_image)).start()
                                last_email_time = time.time()
                                intruder_detected_time = None
                    else:
                        status = "authorized"
                        image_path= ""

                    # Log the detection
                    # ✅ Only log if cooldown passed
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
            
            cv2.imshow('Intrusion Detection', img)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        logger.info("Intrusion Detection stopped")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()
