from flask import request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import re
from pathlib import Path
import logging
import tensorflow as tf
from config import Config
from typing import Dict, List, Optional, Union, Any

# Configure logging
route_logger = logging.getLogger(__name__)

DISEASE_CLASSES = [
    'Normal',
    'Cataract',
    'Glaucoma',
    'Diabetic Retinopathy'
]

def init_routes(app: Any, db: Any) -> Any:
    users_collection = db['users']
    patient_history_collection = db['patient_history']

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    def validate_email(email: str) -> bool:
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None

    def preprocess_image(filepath: Union[str, Path]) -> Optional[np.ndarray]:
        try:
            if not os.path.exists(filepath):
                route_logger.error(f"File not found: {filepath}")
                return None

            with Image.open(filepath) as img:
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                return np.expand_dims(img_array, axis=0)
        except Exception as e:
            route_logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def get_prediction(preprocessed_image: np.ndarray) -> str:
        try:
            if current_app.model is None:
                raise ValueError("Model is not loaded.")

            predictions = current_app.model.predict(preprocessed_image)

            if not isinstance(predictions, np.ndarray):
                route_logger.error(f"Unexpected prediction type: {type(predictions)}")
                raise ValueError("Model prediction has unexpected type")

            if len(predictions.shape) != 2 or predictions.shape[1] != len(DISEASE_CLASSES):
                route_logger.error(f"Unexpected prediction shape: {predictions.shape}")
                raise ValueError("Model prediction has unexpected shape")

            predicted_class_index = np.argmax(predictions[0])
            predicted_disease = DISEASE_CLASSES[predicted_class_index]

            route_logger.info(f"Prediction successful: {predicted_disease}")

            return predicted_disease
        except Exception as e:
            route_logger.error(f"Error during prediction: {str(e)}")
            raise

    @app.route('/test_upload', methods=['POST'])
    def test_upload():
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = Path(app.config['UPLOAD_FOLDER']) / filename

                route_logger.info(f"Attempting to save file to: {filepath}")
                file.save(filepath)

                file_exists = filepath.exists()
                file_size = filepath.stat().st_size if file_exists else 0

                return jsonify({
                    'success': True,
                    'filename': filename,
                    'filepath': str(filepath),
                    'file_exists': file_exists,
                    'file_size': file_size
                }), 200
            else:
                return jsonify({'error': 'File type not allowed'}), 400
        except Exception as e:
            route_logger.error(f"Error in test upload: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        try:
            # Secure and save the file
            filename = secure_filename(file.filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename

            # Save the file using a string path for compatibility
            file.save(str(filepath))

            # Preprocess the image
            preprocessed_image = preprocess_image(filepath)
            if preprocessed_image is None:
                return jsonify({'error': 'Error processing image'}), 500

            # Perform prediction
            predicted_disease = get_prediction(preprocessed_image)

            return jsonify({
                'success': True,
                'prediction': predicted_disease
            }), 200

        except Exception as e:
            # Log the error during the prediction process
            route_logger.error(f"Error during prediction process: {str(e)}")
            return jsonify({
                'error': 'Internal server error during prediction',
                'details': str(e)
            }), 500

        finally:
            if 'filepath' in locals():
                try:
                    # Remove the file after processing
                    os.remove(filepath)
                except Exception as e:
                    # Log any errors that occur during file removal
                    route_logger.error(f"Error removing temporary file: {str(e)}")

    @app.route('/signup', methods=['POST'])
    def signup():
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not all([username, email, password]):
            return jsonify({"error": "All fields are required"}), 400

        if not validate_email(email):
            return jsonify({"error": "Invalid email format"}), 400

        if users_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
            return jsonify({"error": "Username or email already exists"}), 400

        try:
            hashed_password = generate_password_hash(password)
            user_id = users_collection.insert_one({
                "username": username,
                "email": email,
                "password": hashed_password
            }).inserted_id

            return jsonify({
                "message": "User created successfully",
                "user_id": str(user_id)
            }), 201
        except Exception as e:
            route_logger.error(f"Error during signup: {str(e)}")
            return jsonify({"error": "Error creating user"}), 500

    @app.route('/login', methods=['POST'])
    def login():
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not all([username, password]):
            return jsonify({"error": "Username and password are required"}), 400

        try:
            user = users_collection.find_one({"username": username})
            if user and check_password_hash(user['password'], password):
                return jsonify({
                    "message": "Login successful",
                    "user_id": str(user['_id'])
                }), 200

            return jsonify({"error": "Invalid credentials"}), 401
        except Exception as e:
            route_logger.error(f"Error during login: {str(e)}")
            return jsonify({"error": "Error during login"}), 500

    @app.route('/patient_history/<patient_id>', methods=['GET'])
    def get_patient_history(patient_id: str):
        try:
            history = list(patient_history_collection.find({'patient_id': patient_id}))
            for item in history:
                item['_id'] = str(item['_id'])
            return jsonify(history), 200
        except Exception as e:
            route_logger.error(f"Error fetching patient history: {str(e)}")
            return jsonify({'error': 'Error fetching patient history'}), 500

    return app
