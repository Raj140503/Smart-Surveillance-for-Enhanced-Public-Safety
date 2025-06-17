import cv2
import numpy as np
import time
from tracker import ObjectTracker
from email_alert import send_alert_email_async  # Use the async email function

tracker = ObjectTracker()
last_email_time = 0  # Variable to track last email sent time

def process_video(source):
    """Process video frames and detect abandoned objects."""
    cap = cv2.VideoCapture(source)

    # Reduce buffering issues
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, firstframe = cap.read()
    if not ret:
        print("Error: Unable to read from video source.")
        return

    firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    firstframe_blur = cv2.GaussianBlur(firstframe_gray, (3, 3), 0)

    global last_email_time

    frame_count = 0  # Counter to skip frames
    frame_skip = 1   # Process every 2nd frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip processing this frame

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (3, 3), 0)
        frame_diff = cv2.absdiff(firstframe_blur, frame_blur)

        edged = cv2.Canny(frame_diff, 5, 200)
        kernel = np.ones((10, 10), np.uint8)
        thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in cnts:
            contourArea = cv2.contourArea(c)
            if 50 < contourArea < 10000:
                (x, y, w, h) = cv2.boundingRect(c)
                detections.append([x, y, w, h])

        _, abandoned_objects = tracker.update(detections)

        for obj in abandoned_objects:
            _, x2, y2, w2, h2, _ = obj
            cv2.putText(frame, "Suspicious object detected", (x2, y2 - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

            # Save the image when a suspicious object is detected
            image_path = "suspicious_object.jpg"
            cv2.imwrite(image_path, frame)

            # Send an email alert only once every 30 seconds
            current_time = time.time()
            if current_time - last_email_time > 30:  # 30 seconds cooldown
                send_alert_email_async(image_path)  # Send email in a non-blocking way
                last_email_time = current_time  # Update the last email sent time

        # Convert frame to JPEG and send it for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
