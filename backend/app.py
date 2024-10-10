from pathlib import Path
import tensorflow as tf
import numpy as np
from flask import Flask
from config import Config
from routes import init_routes
import logging
from flask_cors import CORS
from pymongo import MongoClient
from flask import jsonify

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize config
    Config.init_app()

    app = Flask(__name__)
    app.config.from_object('config.Config')
    CORS(app)

    # MongoDB connection
    try:
        client = MongoClient(app.config['MONGO_URI'])
        db = client.get_default_database()
        logger.info("MongoDB connection established successfully")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        db = None

    @app.route('/db_error')
    def db_error():
        return jsonify({"error": "Could not connect to MongoDB."}), 500

    # Load model and verify it works
    try:
        if not Config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

        app.model = tf.keras.models.load_model(app.config['MODEL_PATH'])
        logger.info(f"Model loaded successfully from {app.config['MODEL_PATH']}")

        # Verify model can make predictions
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = app.model.predict(dummy_input)
        logger.info("Model prediction test successful")

    except Exception as e:
        logger.error(f"Error loading or testing model: {str(e)}")
        app.model = None

    @app.route('/model_error')
    def model_error():
        return jsonify({"error": "Could not load the model."}), 500

    # Initialize routes
    init_routes(app, db)

    return app


if __name__ == '__main__':
    application = create_app()
    application.run(debug=application.config['DEBUG'])