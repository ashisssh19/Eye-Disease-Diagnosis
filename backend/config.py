import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/eye_disease_diagnosis')

    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MODEL_PATH = BASE_DIR / 'model' / 'model.h5'
    DEBUG = False

    @classmethod
    def init_app(cls):
        try:
            cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
            cls.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            if not os.getenv('SECRET_KEY'):
                print("Warning: SECRET_KEY is not set. Using default key.")
            if not os.getenv('MONGO_URI'):
                print("Warning: MONGO_URI is not set. Using default MongoDB URI.")

            # Verify model exists
            if not cls.MODEL_PATH.exists():
                print(f"Warning: Model not found at {cls.MODEL_PATH}")

            return True
        except Exception as e:
            print(f"Error initializing app config: {str(e)}")
            return False