import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/eye_disease_diagnosis')
    UPLOAD_FOLDER = BASE_DIR / 'backend' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MODEL_PATH = BASE_DIR / 'model' / 'model.h5'
    DEBUG = False

    @classmethod
    def init_app(cls):
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

