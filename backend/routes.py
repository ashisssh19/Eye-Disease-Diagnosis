from flask import request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from bson import ObjectId
import re


def init_routes(app, db):
    users_collection = db['users']
    patient_history_collection = db['patient_history']

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    def validate_email(email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None

    def preprocess_image(filepath):
        try:
            img = Image.open(filepath)
            img = img.resize((224, 224))  # Adjust size according to your model's input
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            app.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    @app.route('/signup', methods=['POST'])
    def signup():
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            return jsonify({"error": "All fields are required"}), 400

        if not validate_email(email):
            return jsonify({"error": "Invalid email format"}), 400

        if users_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
            return jsonify({"error": "Username or email already exists"}), 400

        hashed_password = generate_password_hash(password)
        user_id = users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password
        }).inserted_id
        return jsonify({"message": "User created successfully", "user_id": str(user_id)}), 201

    @app.route('/login', methods=['POST'])
    def login():
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user['password'], password):
            return jsonify({"message": "Login successful", "user_id": str(user['_id'])}), 200
        return jsonify({"error": "Invalid credentials"}), 401

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            preprocessed_image = preprocess_image(filepath)
            if preprocessed_image is None:
                return jsonify({'error': 'Error processing image'}), 500

            # Make prediction
            # TODO: Implement actual model prediction
            # prediction = model.predict(preprocessed_image)

            # For now, let's return a dummy result
            prediction = {'disease': 'cataract', 'confidence': 0.85}

            # Save to patient history
            patient_id = request.form.get('patient_id')
            if not patient_id:
                return jsonify({'error': 'Patient ID is required'}), 400

            patient_history_collection.insert_one({
                'patient_id': patient_id,
                'filename': filename,
                'prediction': prediction
            })

            return jsonify(prediction), 200
        return jsonify({'error': 'File type not allowed'}), 400

    @app.route('/patient_history/<patient_id>', methods=['GET'])
    def get_patient_history(patient_id):
        if not patient_id:
            return jsonify({'error': 'Patient ID is required'}), 400

        history = list(patient_history_collection.find({'patient_id': patient_id}))
        for item in history:
            item['_id'] = str(item['_id'])  # Convert ObjectId to string
        return jsonify(history), 200

    return app