import logging
import numpy as np
from src.ml.model_manager import ModelManager
from src.core.device_manager import DeviceManager

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Wrapper for YOLOv8 Face Detection.
    """
    def __init__(self):
        self.model_manager = ModelManager()
        self.device_manager = DeviceManager()
        self.model = None

    def load(self):
        if not self.model:
            self.model = self.model_manager.load_model("yolov8-face")

    def detect(self, image_array: np.ndarray):
        """
        Detect faces in an image.
        Returns list of results.
        """
        self.load()
        if not self.model:
            logger.error("Model failed to load.")
            return []

        # Run inference
        # YOLOv8 handles numpy arrays directly (HWC, BGR or RGB)
        # We assume image_array is RGB from our ImageLoader
        
        # Ultralytics track/predict
        results = self.model.predict(
            image_array, 
            device=self.device_manager.get_device(),
            conf=0.5, # Confidence threshold
            verbose=False
        )
        
        # Parse results
        detections = []
        for r in results:
            boxes = r.boxes

            
            # Better iteration for boxes and keypoints together
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                keypoints = r.keypoints.xy.cpu().numpy() # [N, 5, 2]
            else:
                keypoints = None

            for i, box in enumerate(boxes):
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                landmarks = keypoints[i] if keypoints is not None and len(keypoints) > i else None
                
                detections.append({
                    "box": coords,
                    "confidence": conf,
                    "class": cls,
                    "landmarks": landmarks
                })
        
        return detections
