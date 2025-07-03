import smtplib
import ssl
import json
import os
from email.message import EmailMessage
import cv2

def load_email_settings(config_file='config.json'):
    """ Load email settings from a JSON config file. """
    try:
        if not os.path.exists(config_file):
            raise FileNotFoundError("Email configuration file not found.")
        
        with open(config_file, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading email settings: {e}")
        return None

def send_alert(email_settings, full_image_path, intruder_image_path=None):
    """ Send an email alert with one or two image attachments. """
    try:
        msg = EmailMessage()
        msg['Subject'] = "Security Alert: Unauthorized Person Detected"
        msg['From'] = email_settings['Email_Send']
        msg['To'] = email_settings['Email_Receive']
        msg.set_content("An unauthorized person has been detected. See the attached images for details.")
        
        # Attach full-screen image
        with open(full_image_path, 'rb') as img:
            msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename='full_screen.jpg')
        
        # Attach intruder close-up image if provided
        if intruder_image_path:
            with open(intruder_image_path, 'rb') as img:
                msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename='intruder.jpg')
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(email_settings['Email_Send'], email_settings['Email_Password'])
            server.send_message(msg)
        
        print("Alert email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def capture_intruder(image_path, frame, x, y, w, h):
    """ Capture and save an image of the unauthorized person. """
    try:
        intruder_img = frame[y:y+h, x:x+w]
        cv2.imwrite(image_path, intruder_img)
        print("Intruder image saved.")
    except Exception as e:
        print(f"Error capturing intruder image: {e}")
