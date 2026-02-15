
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging
import numpy as np
from src.core.device_manager import DeviceManager

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """
    Extracts semantic embeddings using CLIP.
    """
    def __init__(self):
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device()
        self.model = None
        self.processor = None
        self.model_name = "openai/clip-vit-base-patch32"

    def load(self):
        if self.model is None:
            try:
                logger.info(f"Loading CLIP ({self.model_name})...")
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
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
            
            # Normalize
            embedding = outputs.cpu().numpy().flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting semantic embedding: {e}")
            return None
