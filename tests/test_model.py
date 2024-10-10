
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import Config
from backend.utils import load_model, predict


def test_model():
    try:
        # Initialize config
        Config.init_app()

        # Test model loading
        model = load_model()
        logger.info("Model loaded successfully")

        # Test prediction with a test image
        test_image_path = Config.MODEL_PATH.parent / '1.jpg'
        if test_image_path.exists():
            result = predict(test_image_path)
            logger.info(f"Test prediction result: {result}")
        else:
            logger.warning(f"Test image not found at {test_image_path}")

        return True
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_model()
    logger.info(f"Model test {'successful' if success else 'failed'}")
