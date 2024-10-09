import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import Config

def test_model():
    try:
        # Initialize config and make sure directories exist
        Config.init_app()

        # Load model
        model_path = Config.MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")

        # Test prediction with dummy input
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        prediction = model.predict(dummy_input)
        print(f"Model prediction shape: {prediction.shape}")

        # Load and preprocess a test image if available
        test_image_path = model_path.parent / 'test_image.jpg'
        if test_image_path.exists():
            with Image.open(test_image_path) as img:
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

            # Make prediction with the image
            test_prediction = model.predict(img_array)
            print(f"Test image prediction shape: {test_prediction.shape}")
        else:
            print(f"Test image not found at {test_image_path}")

        return True
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model()
    print(f"Model test {'successful' if success else 'failed'}")
