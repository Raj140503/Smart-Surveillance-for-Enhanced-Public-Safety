from flask import Flask, render_template, Response, request, redirect, url_for
import os
import cv2
import abandoned_object_detection  # Import detection script
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create an upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}

def allowed_file(filename):
    """Check if the file is a valid video format."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file uploads."""
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        return redirect(url_for("video_feed", source=filepath))

    return "Invalid file type", 400

@app.route("/video_feed")
def video_feed():
    """Stream video based on selected input source."""
    source = request.args.get("source", "")

    if source == "webcam":
        return Response(abandoned_object_detection.process_video(0),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    elif os.path.exists(source):  # Check if file exists
        return Response(abandoned_object_detection.process_video(source),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    else:
        return "No valid video source provided.", 400

if __name__ == "__main__":
    app.run(port=5003,debug=True)
