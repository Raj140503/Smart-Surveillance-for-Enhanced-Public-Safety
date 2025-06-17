from flask import Flask, request, render_template, redirect, url_for
import subprocess
import os

app = Flask(__name__)

# Define paths for prototxt and model
PROTOTXT_PATH = "detector/MobileNetSSD_deploy.prototxt"
MODEL_PATH = "detector/MobileNetSSD_deploy.caffemodel"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_video', methods=['POST'])
def run_video():
    video_file = request.form.get('video_file')

    if not video_file:
        return "Error: No video file provided. Please specify a file."

    command = [
        "python", "people_counter.py", 
        "--prototxt", PROTOTXT_PATH, 
        "--model", MODEL_PATH, 
        "--input", video_file
    ]

    # Run the command and stream the output to the console
    subprocess.run(command)

    return "Video processing completed. Check the console for results."

@app.route('/run_webcam', methods=['POST'])
def run_webcam():
    command = [
        "python", "people_counter.py", 
        "--prototxt", PROTOTXT_PATH, 
        "--model", MODEL_PATH
    ]

    # Run the command
    subprocess.run(command)

    return "Webcam processing completed. Check the console for results."

if __name__ == '__main__':
    # Ensure the prototxt and model files exist
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
        print("Error: Missing prototxt or model file.")
        exit(1)

    app.run(port=5002, debug=True)
