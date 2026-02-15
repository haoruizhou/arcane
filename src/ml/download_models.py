
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import CLIPProcessor, CLIPModel

def download_all():
    mm = ModelManager()
    logger.info(f"Models directory: {mm.models_dir}")
    
    # YOLOv8 Face
    for model_key in mm.MODELS:
        if not mm.is_model_downloaded(model_key):
            logger.info(f"Downloading {model_key}...")
            mm.download_model(model_key)
        else:
            logger.info(f"Model {model_key} already downloaded.")
            
    # CLIP
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_path = os.path.join(mm.models_dir, "clip-vit-base-patch32")
    
    if not os.path.exists(clip_path):
        logger.info(f"Downloading {clip_model_name} to {clip_path}...")
        try:
            model = CLIPModel.from_pretrained(clip_model_name)
            processor = CLIPProcessor.from_pretrained(clip_model_name)
            
            model.save_pretrained(clip_path)
            processor.save_pretrained(clip_path)
            logger.info("CLIP model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download CLIP model: {e}")
    else:
        logger.info(f"CLIP model already exists at {clip_path}")

if __name__ == "__main__":
    download_all()
