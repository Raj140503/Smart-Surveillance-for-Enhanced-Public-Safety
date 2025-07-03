from flask import Flask, request, jsonify, render_template, send_from_directory
import json
import os
import subprocess
import shutil
from werkzeug.exceptions import NotFound

app = Flask(__name__)

# Configuration
NAMES_FILE = 'names.json'
IMAGE_DIR = 'images'
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Helper functions
def get_users():
    """Load users from names.json"""
    if not os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, 'w') as f:
            json.dump({}, f)
        return {}
    
    try:
        with open(NAMES_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_users(users):
    """Save users to names.json"""
    with open(NAMES_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def get_next_id():
    """Get next available user ID"""
    users = get_users()
    if not users:
        return 0
    return max(map(int, users.keys())) + 1

def get_user_images(username):
    """Get list of images for a user"""
    user_dir = os.path.join(IMAGE_DIR, username)
    if not os.path.exists(user_dir):
        return []
    
    return [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Routes
@app.route('/')
def index():
    return render_template('database.html')

@app.route('/api/people', methods=['GET'])
def get_people():
    """Get all registered users"""
    return jsonify(get_users())

@app.route('/api/add_person', methods=['POST'])
def add_person():
    """Add a new person and take photos"""
    data = request.get_json()
    if not data or 'name' not in data:
        return "Name is required", 400
    
    name = data['name'].strip()
    if not name:
        return "Name cannot be empty", 400
    
    # Run face_taker.py to capture images
    try:
        # Create a process to run face_taker.py
        # We're using a technique that allows us to "input" the name to the script
        process = subprocess.Popen(['python', 'face_taker.py'], 
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Send the name to the script
        stdout, stderr = process.communicate(input=f"{name}\n")
        
        if process.returncode != 0:
            return f"Error capturing images: {stderr}", 500
        
        # Reload users to get the new ID
        users = get_users()
        user_id = None
        
        for id, username in users.items():
            if username == name:
                user_id = id
                break
        
        if user_id is None:
            return "Failed to add user", 500
            
        return jsonify({"id": user_id, "name": name})
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train the face recognition model"""
    try:
        process = subprocess.run(['python', 'face_trainer.py'], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        
        return jsonify({"message": "Model trained successfully"})
    
    except subprocess.CalledProcessError as e:
        return f"Error training model: {e.stderr}", 500
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/people/<user_id>', methods=['DELETE'])
def delete_person(user_id):
    """Delete a person and their images"""
    users = get_users()
    
    if user_id not in users:
        return "User not found", 404
    
    username = users[user_id]
    user_dir = os.path.join(IMAGE_DIR, username)
    
    # Remove from names.json
    del users[user_id]
    save_users(users)
    
    # Delete user directory 
    if os.path.exists(user_dir):
        try:
            shutil.rmtree(user_dir)
        except Exception as e:
            return f"Error deleting user directory: {str(e)}", 500
    
    return jsonify({"message": "User deleted successfully"})

@app.route('/api/images/<username>', methods=['GET'])
def get_images(username):
    """Get all images for a user"""
    images = get_user_images(username)
    return jsonify(images)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve user images"""
    directory = os.path.dirname(filename)
    file = os.path.basename(filename)
    try:
        return send_from_directory(os.path.join(IMAGE_DIR, directory), file)
    except NotFound:
        return send_from_directory(STATIC_DIR, 'placeholder.png')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(STATIC_DIR, filename)

# Create a placeholder image for users without photos
def create_placeholder():
    placeholder_path = os.path.join(STATIC_DIR, 'placeholder.png')
    if not os.path.exists(placeholder_path):
        try:
            # Create a simple placeholder image using PIL if available
            from PIL import Image, ImageDraw
            
            img = Image.new('RGB', (100, 100), color=(200, 200, 200))
            d = ImageDraw.Draw(img)
            d.text((35, 40), "No\nImage", fill=(80, 80, 80))
            img.save(placeholder_path)
        except ImportError:
            # If PIL is not available, just create an empty file
            with open(placeholder_path, 'wb') as f:
                f.write(b'')

if __name__ == '__main__':
    create_placeholder()
    app.run(port=5005,debug=True)