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
from typing import Dict, List, Optional, Union, Any

# Configure logging
route_logger = logging.getLogger(__name__)

# Disease classes - update these according to your model's classes
DISEASE_CLASSES = [
    'Normal',
    'Cataract',
    'Glaucoma',
    'Diabetic Retinopathy'
]


def init_routes(app: Any, db: Any) -> Any:
    users_collection = db['users']
    patient_history_collection = db['patient_history']

    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

    def get_prediction(preprocessed_image: np.ndarray) -> Optional[Dict[str, Any]]:
        try:
            predictions = current_app.model.predict(preprocessed_image)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_disease = DISEASE_CLASSES[predicted_class_index]

            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    "disease": DISEASE_CLASSES[idx],
                    "confidence": float(predictions[0][idx])
                }
                for idx in top_3_indices
            ]

            return {
                "primary_prediction": {
                    "disease": predicted_disease,
                    "confidence": confidence
                },
                "all_predictions": top_3_predictions
            }
        except Exception as e:
            route_logger.error(f"Error during prediction: {str(e)}")
            return None

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
            filename = secure_filename(file.filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(filepath)

            preprocessed_image = preprocess_image(filepath)
            if preprocessed_image is None:
                return jsonify({'error': 'Error processing image'}), 500

            prediction_result = get_prediction(preprocessed_image)
            if prediction_result is None:
                return jsonify({'error': 'Error making prediction'}), 500

            patient_id = request.form.get('patient_id')
            if patient_id:
                patient_history_collection.insert_one({
                    'patient_id': patient_id,
                    'filename': filename,
                    'prediction': prediction_result
                })

            return jsonify({
                'success': True,
                'prediction': prediction_result
            }), 200

        except Exception as e:
            route_logger.error(f"Error during prediction process: {str(e)}")
            return jsonify({'error': 'Internal server error during prediction'}), 500

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

    @app.route('/health', methods=['GET'])
    def health_check():
        if not hasattr(current_app, 'model') or current_app.model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        return jsonify({
            'status': 'healthy',
            'message': 'Service is running and model is loaded'
        }), 200

    return app