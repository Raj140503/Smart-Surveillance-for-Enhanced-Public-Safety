from flask import Flask, render_template, Response, request, jsonify
import cv2
import cvzone
import json
import threading
import time
from ultralytics import YOLO
from email_alert import send_email_with_image

app = Flask(__name__)

# Load configuration
with open("CONFIG.json", "r") as config_file:
    config = json.load(config_file)
ALERT_ENABLED = config["ALERT"]

# Load YOLO model (Using YOLOv8s for improved accuracy)
model = YOLO('yolov8s.pt')

# Load class names
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Alert cooldown variables
last_alert_time = 0
alert_cooldown = 30  # seconds

# Global variable to track fall status
fall_detected = False

def process_frame(frame):
    global last_alert_time, fall_detected
    results = model(frame, verbose=False)
    
    # Reset fall detection status at the beginning of each frame processing
    current_frame_fall = False
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_detect = int(box.cls[0].item())
            
            # Process only high-confidence person detections
            if confidence > 0.8 and classnames[class_detect] == 'person':
                width, height = x2 - x1, y2 - y1
                scale = 1.1  # Increase bounding box by 10%
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                new_width = int(width * scale)
                new_height = int(height * scale)
                new_x1 = center_x - new_width // 2
                new_y1 = center_y - new_height // 2
                
                # Compute a threshold from scaled dimensions
                threshold = new_height - new_width
                
                # Draw the bounding box and label
                cvzone.cornerRect(frame, [new_x1, new_y1, new_width, new_height], l=20, rt=4)
                cvzone.putTextRect(frame, f'{classnames[class_detect]}', [new_x1 + 5, new_y1 - 10], thickness=1, scale=1.5)
                
                # Fall detection logic
                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected', [new_x1, new_y1 - 30], thickness=1, scale=1.5)
                    current_frame_fall = True
                    fall_image_path = "fall_detected.jpg"
                    cv2.imwrite(fall_image_path, frame)
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        last_alert_time = current_time
                        if ALERT_ENABLED:
                            threading.Thread(target=send_email_with_image, args=(fall_image_path,)).start()
    
    # Update the global fall detection status
    fall_detected = current_frame_fall
    
    return frame

def generate_frames(source):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FPS, 20)  # reduce FPS for performance
    frame_skip = 2  # adjust if needed
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        frame = cv2.resize(frame, (1280, 720))
        frame = process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live_feed')
def live_feed():
    # Stream live webcam feed (default index 0)
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    # Get video file path from query parameter
    video_path = request.args.get('video_path')
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_fall_status')
def check_fall_status():
    # Endpoint for the frontend to check if a fall was detected
    global fall_detected
    status = {'fall_detected': fall_detected}
    
    # Reset fall detection after reporting to prevent multiple alerts for the same fall
    if fall_detected:
        # Keep the status true for this response but clear it for the next check
        threading.Timer(1.0, lambda: setattr(__builtins__, 'fall_detected', False)).start()
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(port=5001, debug=True)