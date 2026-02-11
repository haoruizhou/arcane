import os
import torch
import logging
import sys
from huggingface_hub import hf_hub_download
from src.core.device_manager import DeviceManager

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the download and loading of ML models.
    """
    MODELS = {
        "yolov8-face": {
            "repo_id": "arnabdhar/YOLOv8-Face-Detection",
            "filename": "model.pt",
            "local_name": "model.pt"
        }
    }

    def __init__(self):
        self.device_manager = DeviceManager()
        
        # Determine models directory
        if getattr(sys, 'frozen', False):
            # If bundled by PyInstaller
            base_path = sys._MEIPASS
            self.models_dir = os.path.join(base_path, "models")
        else:
            # If running from source
            self.models_dir = os.path.join(os.path.dirname(__file__), "../../models")
            
        os.makedirs(self.models_dir, exist_ok=True)
        self.loaded_models = {}

    def is_model_downloaded(self, model_key: str) -> bool:
        """Check if model weights exist locally."""
        if model_key not in self.MODELS:
            return False
        
        path = os.path.join(self.models_dir, self.MODELS[model_key]['local_name'])
        return os.path.exists(path)

    def download_model(self, model_key: str, progress_callback=None):
        """
        Download model weights from HF Hub.
        """
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        model_info = self.MODELS[model_key]
        logger.info(f"Downloading {model_key}...")
        
        try:
            # We can use hf_hub_download to get the path, then copy or symlink,
            # or just use the cache. For simplicity let's rely on HF cache for now
            # but usually we want to copy to our models dir for portability.
            
            # Using cache_dir=self.models_dir might create a messy cache structure.
            # Start with simple download.
            
            file_path = hf_hub_download(
                repo_id=model_info['repo_id'],
                filename=model_info['filename'],
                local_dir=self.models_dir,
                local_dir_use_symlinks=False
            )
            
            if progress_callback:
                progress_callback(1.0)
                
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download model {model_key}: {e}")
            raise

    def load_model(self, model_key: str):
        """
        Load the model into memory.
        """
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
            
        if not self.is_model_downloaded(model_key):
             self.download_model(model_key)
             
        path = os.path.join(self.models_dir, self.MODELS[model_key]['local_name'])
        device = self.device_manager.get_device()
        
        logger.info(f"Loading {model_key} to {device}...")
        try:
            from ultralytics import YOLO
            # Ultralytics maps 'device' string
            model = YOLO(path)
            # model.to(device) # YOLO handles device at inference time usually, or via argument
            self.loaded_models[model_key] = model
            return model
        except ImportError:
            logger.error("Ultralytics not installed. Please install 'ultralytics' package.")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise
