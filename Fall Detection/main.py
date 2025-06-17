import cv2
import cvzone
import math
import json
import threading
import time  # For cooldown timer
from ultralytics import YOLO
from email_alert import send_email_with_image  # Import email function

# Load configuration
with open("CONFIG.json", "r") as config_file:
    config = json.load(config_file)

ALERT_ENABLED = config["ALERT"]

# Load YOLO model (Using YOLOv8s for improved accuracy)
model = YOLO('yolov8s.pt')

# Load class names
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Initialize video capture
cap = cv2.VideoCapture('04.mp4')
cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS for performance

frame_skip = 2  # Process every frame; change to >1 to skip frames
frame_count = 0

# Initialize alert cooldown variables
last_alert_time = 0
alert_cooldown = 30  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames based on the defined frame_skip

    frame = cv2.resize(frame, (1280, 720))  # Resize for faster processing
    results = model(frame, verbose=False)  # Run YOLO detection

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_detect = int(box.cls[0].item())

            if confidence > 0.8 and classnames[class_detect] == 'person':
                # Original bounding box dimensions
                width, height = x2 - x1, y2 - y1

                # Increase bounding box size by a small scale factor (e.g., 10%)
                scale = 1.1  # 10% increase
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                new_width = int(width * scale)
                new_height = int(height * scale)
                new_x1 = center_x - new_width // 2
                new_y1 = center_y - new_height // 2

                # Compute threshold using the scaled dimensions
                threshold = new_height - new_width

                # Draw the scaled bounding box and label
                cvzone.cornerRect(frame, [new_x1, new_y1, new_width, new_height], l=20, rt=4)
                cvzone.putTextRect(frame, f'{classnames[class_detect]}', [new_x1 + 5, new_y1 - 10], thickness=1, scale=1.5)

                # Fall detection logic
                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected', [new_x1, new_y1 - 30], thickness=1, scale=1.5)

                    # Save the frame as an image
                    fall_image_path = "fall_detected.jpg"
                    cv2.imwrite(fall_image_path, frame)

                    # Check if the cooldown period has passed before sending an email
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        last_alert_time = current_time
                        # Send email alert if enabled, using a thread to avoid lag
                        if ALERT_ENABLED:
                            threading.Thread(target=send_email_with_image, args=(fall_image_path,)).start()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
