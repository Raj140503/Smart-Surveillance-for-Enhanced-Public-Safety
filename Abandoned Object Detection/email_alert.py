import smtplib
import json
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email(image_path):
    """Send an email alert with the detected object's image attached."""
    with open("config.json", "r") as file:
        config = json.load(file)

    sender_email = config["Email_Send"]
    receiver_email = config["Email_Receive"]
    sender_password = config["Email_Password"]

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Suspicious Object Detected"

    body = "A suspicious object has been detected. See the attached image."
    msg.attach(MIMEText(body, 'plain'))

    try:
        with open(image_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={image_path}')
            msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("✅ Alert email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def send_alert_email_async(image_path):
    """Run the email sending function in a separate thread."""
    threading.Thread(target=send_email, args=(image_path,)).start()
