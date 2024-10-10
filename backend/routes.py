from flask import request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import re
from pathlib import Path
import logging
import datetime
import tensorflow as tf
from config import Config
from typing import Dict, List, Optional, Union, Any

# Configure logging
route_logger = logging.getLogger(__name__)

DISEASE_CLASSES = [
    'Cataract',
    'Diabetic Retinopathy',
    'Glaucoma',
    'Normal'
]

def init_routes(app: Any, db: Any) -> Any:
    users_collection = db['users']
    patient_history_collection = db['patient_history']

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    def validate_email(email: str) -> bool:
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None

    def preprocess_image(filepath):
        try:
            route_logger.info(f"Preprocessing image at {filepath}")
            image = Image.open(filepath)
            image = image.resize((224, 224))  # Adjusted image resizing
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        except Exception as e:
            route_logger.error(f"Error in preprocess_image: {str(e)}")
            return None

    def get_prediction(preprocessed_image):
        try:
            route_logger.info("Running model prediction...")
            predictions = app.model.predict(preprocessed_image)
            route_logger.info(f"Raw predictions: {predictions}")
            predicted_label = np.argmax(predictions, axis=1)[0]
            route_logger.info(f"Prediction result: {predicted_label}")
            return int(predicted_label)
        except Exception as e:
            route_logger.error(f"Error in get_prediction: {str(e)}")
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
            else:
                return jsonify({'error': 'File type not allowed'}), 400
        except Exception as e:
            route_logger.error(f"Error in test upload: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/predict', methods=['POST'])
    def predict():
        app.logger.info("Predict route accessed")
        app.logger.info(f"Upload folder path: {app.config['UPLOAD_FOLDER']}")

        try:
            if 'file' not in request.files:
                app.logger.error("No file part in request")
                return jsonify({'error': 'No file part'}), 400

            files = request.files.getlist('file')
            app.logger.info(f"Number of files received: {len(files)}")

            patient_id = request.form.get('patient_id')
            app.logger.info(f"Patient ID: {patient_id}")

            if not files or all(file.filename == '' for file in files):
                app.logger.error("No selected files")
                return jsonify({'error': 'No selected files'}), 400

            predictions = []

            for file in files:
                if file and allowed_file(file.filename):
                    try:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        app.logger.info(f"Attempting to save file at: {filepath}")
                        file.save(filepath)
                        app.logger.info(f"File saved successfully at: {filepath}")

                        preprocessed_image = preprocess_image(filepath)
                        if preprocessed_image is None:
                            return jsonify({'error': 'Error preprocessing image'}), 500

                        prediction_index = get_prediction(preprocessed_image)
                        if prediction_index is None:
                            return jsonify({'error': 'Error making prediction'}), 500

                        try:
                            prediction_index = int(prediction_index)
                        except ValueError:
                            app.logger.error(f"Invalid prediction index for {filename}: {prediction_index}")
                            return jsonify({'error': 'Invalid prediction index'}), 500

                        # Commenting out the probabilities part
                        # probabilities = app.model.predict(preprocessed_image)[0]

                        # Create response for the current file
                        prediction_result = {
                            'filename': filename,
                            'disease': DISEASE_CLASSES[prediction_index] if 0 <= prediction_index < len(
                                DISEASE_CLASSES) else 'Unknown',
                            # 'probabilities': {class_name: float(prob) for class_name, prob in zip(DISEASE_CLASSES, probabilities)}  # Commented out
                        }

                        if patient_id and hasattr(app, 'patient_history_collection'):
                            app.patient_history_collection.insert_one({
                                'patient_id': patient_id,
                                'filename': filename,
                                'prediction': prediction_result,
                                'timestamp': datetime.utcnow()
                            })

                        predictions.append(prediction_result)

                    except Exception as e:
                        app.logger.error(f"Error during prediction process for {file.filename}: {str(e)}")
                        predictions.append({'filename': file.filename, 'error': f'Prediction process error: {str(e)}'})

                    finally:
                        try:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                                app.logger.info(f"Cleaned up file: {filepath}")
                            else:
                                app.logger.warning(f"File not found for cleanup: {filepath}")
                        except Exception as e:
                            app.logger.error(f"Error removing temporary file: {str(e)}")

                else:
                    app.logger.error(f"File type not allowed: {file.filename}")
                    predictions.append({'filename': file.filename, 'error': 'File type not allowed'})

            return jsonify(predictions), 200

        except Exception as e:
            app.logger.error(f"Unexpected error in predict route: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred'}), 500

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
        try:
            # Verify model is loaded
            if not hasattr(app, 'model'):
                return jsonify({
                    'status': 'error',
                    'message': 'Model not loaded'
                }), 503

            # Check upload directory exists and is writable
            upload_dir = Path(app.config['UPLOAD_FOLDER'])
            if not upload_dir.exists() or not os.access(upload_dir, os.W_OK):
                return jsonify({
                    'status': 'error',
                    'message': 'Upload directory not accessible'
                }), 503

            # Optional: Check database connection
            if hasattr(app, 'patient_history_collection'):
                app.patient_history_collection.find_one({})

            return jsonify({
                'status': 'healthy',
                'message': 'All systems operational'
            }), 200

        except Exception as e:
            app.logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Health check failed: {str(e)}'
            }), 503

    return app