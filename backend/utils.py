import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from config import Config


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def preprocess_image(filepath, target_size=(224, 224)):
    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB')
            img = img.resize(target_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize pixel values

            # Log image array shape for debugging
            print(f"Processed image shape: {img_array.shape}")

        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None


def get_prediction(model, image):
    try:
        prediction = model.predict(image)

        # Log prediction output for debugging
        print(f"Prediction output: {prediction}")

        return prediction
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
