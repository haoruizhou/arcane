
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_all():
    mm = ModelManager()
    logger.info(f"Models directory: {mm.models_dir}")
    
    for model_key in mm.MODELS:
        if not mm.is_model_downloaded(model_key):
            logger.info(f"Downloading {model_key}...")
            mm.download_model(model_key)
        else:
            logger.info(f"Model {model_key} already downloaded.")

if __name__ == "__main__":
    download_all()
