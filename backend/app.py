from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
from config import Config
from routes import init_routes
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    # MongoDB connection
    client = MongoClient(app.config['MONGO_URI'])
    db = client.get_default_database()

    # Initialize routes
    init_routes(app, db)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=app.config['DEBUG'])