
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging
import numpy as np
from src.core.device_manager import DeviceManager
from src.ml.model_manager import ModelManager

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """
    Extracts semantic embeddings using CLIP.
    """
    def __init__(self):
        self.device_manager = DeviceManager()
        self.model_manager = ModelManager()
        self.device = self.device_manager.get_device()
        self.model = None
        self.processor = None
        self.model_name = "clip-vit-base-patch32"

    def load(self):
        if self.model is None:
            try:
                # Check local models dir
                local_path = os.path.join(self.model_manager.models_dir, "clip-vit-base-patch32")
                
                if os.path.exists(local_path):
                    logger.info(f"Loading CLIP from {local_path}...")
                    self.model = CLIPModel.from_pretrained(local_path).to(self.device).eval()
                    self.processor = CLIPProcessor.from_pretrained(local_path)
                else:
                    logger.warning(f"Local CLIP model not found at {local_path}. trying to download from HF...")
                    model_name = "openai/clip-vit-base-patch32"
                    self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
                    self.processor = CLIPProcessor.from_pretrained(model_name)
                    
                logger.info("CLIP loaded.")
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Returns the CLIP image embedding (512d for base-patch32).
        """
        self.load()
        if self.model is None:
            return None

        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            # transformers 5.x may return BaseModelOutputWithPooling
            if hasattr(outputs, 'image_embeds'):
                embedding = outputs.image_embeds.cpu().numpy().flatten()
            elif hasattr(outputs, 'cpu'):
                embedding = outputs.cpu().numpy().flatten()
            else:
                embedding = outputs.pooler_output.cpu().numpy().flatten()
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting semantic embedding: {e}")
            return None

