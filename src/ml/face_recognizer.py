
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import logging
from src.core.device_manager import DeviceManager

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Generates embeddings for faces using InceptionResnetV1 (FaceNet).
    """
    def __init__(self):
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device()
        self.model = None

    def load(self):
        if self.model is None:
            try:
                logger.info("Loading FaceNet (InceptionResnetV1)...")
                self.model = InceptionResnetV1(pretrained='vggface2').eval()
                self.model.to(self.device)
                logger.info("FaceNet loaded.")
            except Exception as e:
                logger.error(f"Failed to load FaceNet: {e}")

    def get_embedding(self, face_image: Image.Image) -> np.ndarray:
        """
        Returns a 512d numpy array representing the face embedding.
        Input image should be a crop of the face.
        """
        self.load()
        if self.model is None:
            return None

        try:
            # Preprocess
            # facenet-pytorch expects standardized input?
            # It usually handles it, but let's resize to 160x160 as per standard FaceNet
            img = face_image.resize((160, 160))
            
            # Convert to tensor and normalize
            # Standard normalization for FaceNet is usually done by the model if strictly following their pipeline,
            # but InceptionResnetV1 in this repo expects:
            # "whitening" or specific standardization. 
            # Let's use standard ToTensor and rely on the model or manual whitening.
            # actually facenet-pytorch training usually uses fixed_image_standardization.
            
            img_np = np.array(img).astype(np.float32)
            # Whiten
            mean = img_np.mean()
            std = img_np.std()
            img_normalized = (img_np - mean) / max(std, 1e-5)
            
            # To Tensor [C, H, W]
            img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1))
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(img_tensor)
                
            return embedding.cpu().detach().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None
