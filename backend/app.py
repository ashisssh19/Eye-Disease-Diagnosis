from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
import tensorflow as tf
import os
import logging
from config import Config


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

    # Load model
    try:
        app.model = tf.keras.models.load_model(app.config['MODEL_PATH'])
        logger.info(f"Model loaded successfully from {app.config['MODEL_PATH']}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        app.model = None

    # Initialize routes
    from routes import init_routes
    init_routes(app, db)

    return app


if __name__ == '__main__':
    application = create_app()
    application.run(debug=application.config['DEBUG'])