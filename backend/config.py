import os
from pathlib import Path


class Config:
    # Base directory of the application (backend folder)
    BASE_DIR = Path(__file__).parent

    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'

    # MongoDB connection URI
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/eye_disease_diagnosis'

    # Upload folder configuration
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # Model configuration
    MODEL_PATH = BASE_DIR / 'model' / 'vgg16_fp_scans_final_model_finetuned.keras'

    # Debug mode
    DEBUG = False  # Set to False in production

    @classmethod
    def init_app(cls):
        # Ensure critical directories exist
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.MODEL_PATH.parent, exist_ok=True)