import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FocusDetector:
    """
    Analyzes image sharpness/focus.
    """
    
    @staticmethod
    def measure_sharpness(image_array: np.ndarray) -> float:
        """
        Compute the variance of the Laplacian.
        Higher values indicates more edges/sharpness.
        
        Args:
            image_array: RGB numpy array or Grayscale
        """
        try:
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
                
            # Compute Laplacian
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var
            
        except Exception as e:
            logger.error(f"Error measuring sharpness: {e}")
            return 0.0

    @staticmethod
    def check_face_focus(image_array: np.ndarray, face_box: tuple) -> float:
        """
        Crop face region and check sharpness.
        face_box: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = map(int, face_box)
        
        # Clamp coordinates
        h, w = image_array.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        face_crop = image_array[y1:y2, x1:x2]
        return FocusDetector.measure_sharpness(face_crop)
