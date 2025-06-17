import smtplib
import ssl
import json
from email.message import EmailMessage

# Load configuration from CONFIG.json
with open("CONFIG.json", "r") as config_file:
    config = json.load(config_file)

EMAIL_SENDER = config["Email_Send"]
EMAIL_RECEIVER = config["Email_Receive"]
EMAIL_PASSWORD = config["Email_Password"]

def send_email_with_image(image_path):
    """Send an email with the fall-detection image attached."""
    msg = EmailMessage()
    msg["Subject"] = "Fall Detected Alert"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("A fall has been detected. Please check the attached image.")

    # Attach the image
    with open(image_path, "rb") as img_file:
        msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename="fall_detected.jpg")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")