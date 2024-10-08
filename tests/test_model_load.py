import tensorflow as tf
from backend import Config

def test_load_model():
    try:
        model = tf.keras.models.load_model(Config.MODEL_PATH)
        print(f"Model loaded successfully from {Config.MODEL_PATH}")
        print(f"Model summary:")
        model.summary()
    except Exception as e:
        print(f"Error loading model from {Config.MODEL_PATH}: {str(e)}")

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    test_load_model()