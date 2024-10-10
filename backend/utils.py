import logging
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable for model
model = None


def load_model():
    """Load the model into global variable"""
    global model
    try:
        if model is None:
            if not Config.MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")
            model = tf.keras.models.load_model(Config.MODEL_PATH)
            logger.info(f"Model loaded successfully from {Config.MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def preprocess_image(image_path):
    """Preprocess the image for model prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))  # Adjust size according to your model's requirements
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


def predict(image_path):
    """Make prediction on the image"""
    try:
        # Load model if not already loaded
        model = load_model()

        # Preprocess image
        processed_image = preprocess_image(image_path)

        # Make prediction
        prediction = model.predict(processed_image)

        # Get prediction class and confidence
        class_names = ['Normal', 'Cataract', 'Glaucoma', 'Retina Disease']  # Adjust according to your classes
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise


def is_allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def save_upload_file(file, filename):
    """Save uploaded file to the upload folder"""
    try:
        filepath = Config.UPLOAD_FOLDER / filename
        file.save(str(filepath))
        return filepath
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise