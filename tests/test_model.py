import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import Config


def test_model():
    try:
        # Initialize config
        Config.init_app()

        # Load model
        model = tf.keras.models.load_model(Config.MODEL_PATH)
        print(f"Model loaded successfully from {Config.MODEL_PATH}")

        # Test prediction with dummy input
        dummy_input = np.random.rand(1, 224, 224, 3)
        prediction = model.predict(dummy_input)
        print(f"Model prediction shape: {prediction.shape}")

        # Load and preprocess a test image if available
        test_image_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'test_image.jpg')
        if os.path.exists(test_image_path):
            # Open and convert image to numpy array
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