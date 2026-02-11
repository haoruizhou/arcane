import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging
import numpy as np
from src.core.device_manager import DeviceManager

logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    """
    Extracts feature embeddings from images using a pre-trained CNN (MobileNetV3).
    """
    def __init__(self):
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device()
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load(self):
        if self.model is None:
            try:
                logger.info("Loading MobileNetV3 for embeddings...")
                # Use MobileNetV3 Small for speed
                full_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                
                # We want embeddings, not classification.
                # MobileNetV3 classifier is a Sequential. We can remove the last linear layer
                # or just use the backbone + pooling.
                # Structure: features -> avgpool -> classifier
                # Let's simple remove the last layer of classifier or just use features+pool.
                # Converting to feature extractor:
                full_model.classifier = torch.nn.Identity() # Replace classifier with Identity? 
                # calculated: MobileNetV3 Small avgpool output is 576. 
                # But notice limits of Identity replacement if `classifier` had non-linearities.
                # MobileNetV3 classifier: Linear -> HardSwish -> Dropout -> Linear
                # Check structure.
                
                # Safer: Use `create_feature_extractor` or just manual forward.
                # Let's just use the backbone + avgpool.
                # Actually, simple way:
                self.model = full_model
                self.model.to(self.device)
                self.model.eval()
                logger.info("MobileNetV3 loaded.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")

    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Returns a 1D numpy array representing the image embedding.
        """
        self.load()
        if self.model is None:
            return None

        try:
            # Preprocess
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            img_t = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # MobileNetV3 forward does features -> avgpool -> classifier
                # Since we replaced classifier with Identity, we get the flattened pooled features?
                # Wait, Identity on a [N, 576, 1, 1] might flatten? No.
                # Let's verify output shape or force flatten.
                output = self.model(img_t)
                
            # Flatten and normalize
            embedding = output.cpu().numpy().flatten()
            
            # L2 Normalize (Cosine Similarity requires normalized vectors for simple dot product)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
