import os

class Config:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MONGO_URI = 'mongodb://localhost:27017/eye_disease_diagnosis'
    DEBUG = True