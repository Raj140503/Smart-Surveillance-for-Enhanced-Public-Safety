#face_trainer.py
# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
from PIL import Image
import os
import logging
from settings.settings import PATHS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import json

def generate_names_json(image_dir='images', names_file='names.json'):
    folders = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    names = {str(i): name for i, name in enumerate(folders)}

    with open(names_file, 'w') as f:
        json.dump(names, f, indent=4)
    
    print(f"[INFO] names.json updated with {len(names)} entries.")


def get_images_and_labels(path: str):
    faceSamples = []
    ids = []

    label_map = {}
    label_counter = 0

    for name in sorted(os.listdir(path)):
        user_folder = os.path.join(path, name)
        if not os.path.isdir(user_folder):
            continue

        if name not in label_map:
            label_map[name] = label_counter
            label_counter += 1

        label = label_map[name]

        for filename in os.listdir(user_folder):
            img_path = os.path.join(user_folder, filename)

            # Convert to grayscale
            PIL_img = Image.open(img_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Detect face
            detector = cv2.CascadeClassifier(PATHS['cascade_file'])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(label)

    # Save labels to names.json
    with open(PATHS['names_file'], 'w') as f:
        json.dump({str(v): k for k, v in label_map.items()}, f, indent=4)

    return faceSamples, ids


if __name__ == "__main__":
    try:
        logger.info("Starting face recognition training...")
        
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer.create()
        
        # Get training data
        faces, ids = get_images_and_labels(PATHS['image_dir'])
        
        if not faces or not ids:
            raise ValueError("No training data found")
            
        # Train the model
        logger.info("Training model...")
        recognizer.train(faces, np.array(ids))
        
        # Save the model
        recognizer.write(PATHS['trainer_file'])
        logger.info(f"Model trained with {len(np.unique(ids))} faces")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
