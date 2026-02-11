import numpy as np
import logging
from PIL import Image
from src.core.image_loader import ImageLoader
from src.ml.face_detector import FaceDetector
from src.ml.focus_detector import FocusDetector
from src.ml.eye_detector import EyeOpennessDetector

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """
    Encapsulates the ML pipeline for analyzing images.
    """
    def __init__(self):
        self.face_detector = FaceDetector()
        
    def analyze(self, image_path: str, thumbnail_size: tuple[int, int] = (240, 240)) -> dict:
        """
        Loads a preview of the image and runs ML analysis.
        Returns a dictionary with results.
        """
        result = {
            "path": image_path,
            "thumbnail": None, # PIL Image for UI
            "detections": [],
            "focus_score": 0.0,
            "overall_score": 0.0,
            "error": None
        }
        
        try:
            # Load a decent sized preview for ML (e.g., 640px)
            # We don't want it too small or facial features are lost.
            ml_preview_size = (640, 640)
            img = ImageLoader.load_preview(image_path, max_size=ml_preview_size)
            
            if img is None:
                result["error"] = "Could not load preview"
                return result

            # Create the specific thumbnail for UI
            thumb = img.copy()
            thumb.thumbnail(thumbnail_size)
            result["thumbnail"] = thumb
            
            # Prepare for ML
            img_np = np.array(img)
            
            # Detect Faces
            detections = self.face_detector.detect(img_np)
            
            max_focus = 0.0
            
            # Analyze Findings
            for det in detections:
                box = det['box']
                # Check Focus
                # Note: Focus score depends on resolution, so consistent ML size is important
                sharpness = FocusDetector.check_face_focus(img_np, box)
                det['sharpness'] = sharpness
                if sharpness > max_focus:
                    max_focus = sharpness
                
                # Check Eyes
                if det.get('landmarks') is not None:
                    eye_status = EyeOpennessDetector.check_eyes(img_np, det['landmarks'])
                    det['eye_status'] = eye_status
            
            result["detections"] = detections
            
            if not detections:
                # If no faces, check global sharpness
                global_sharpness = FocusDetector.measure_sharpness(img_np)
                result["focus_score"] = global_sharpness
            else:
                result["focus_score"] = max_focus
            
            result["overall_score"] = result["focus_score"] 
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            result["error"] = str(e)
            
        return result
