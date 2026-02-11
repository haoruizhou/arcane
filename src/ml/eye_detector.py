import numpy as np
import logging

logger = logging.getLogger(__name__)

class EyeOpennessDetector:
    """
    Estimates if eyes are open.
    Requires landmarks (specifically eye centers).
    """
    
    @staticmethod
    def check_eyes(image_array: np.ndarray, landmarks: np.ndarray) -> str:
        """
        Check if eyes are open based on landmarks.
        landmarks: shape (5, 2) -> [LeftEye, RightEye, Nose, LeftMouth, RightMouth]
        
        Note: The 5-point landmarks from YOLOv8-Face are usually just the center of the pupil/eye.
        They don't give eyelid positions, so we cannot calculate Aspect Ratio (EAR).
        
        To truly check openness, we would need to crop the eye region and run a secondary classifier,
        or use a 68-point landmark detector.
        
        For this MVP, we will return 'N/A' to indicate we need a better model, 
        or we could check simple brightness/variance in the eye crop.
        """
        if landmarks is None or len(landmarks) < 2:
            return "Unknown"
            
        # Landmarks: 0=LeftEye, 1=RightEye
        # Let's crop patches around the eyes and check variance (similar to focus).
        # Closed eyes often have less high-frequency detail than open eyes (lashes+pupil+whites).
        
        left_eye_score = EyeOpennessDetector._analyze_eye_patch(image_array, landmarks[0])
        right_eye_score = EyeOpennessDetector._analyze_eye_patch(image_array, landmarks[1])
        
        # Heuristic threshold
        # This is very rough.
        avg_score = (left_eye_score + right_eye_score) / 2.0
        
        if avg_score > 50.0: # Arbitrary threshold
            return "Open"
        else:
            return "Closed?"

    @staticmethod
    def _analyze_eye_patch(image_array: np.ndarray, point: np.ndarray, radius=15) -> float:
        x, y = int(point[0]), int(point[1])
        h, w = image_array.shape[:2]
        
        x1, y1 = max(0, x-radius), max(0, y-radius)
        x2, y2 = min(w, x+radius), min(h, y+radius)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        patch = image_array[y1:y2, x1:x2]
        # Calculate Laplacian variance (sharpness/detail)
        import cv2
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        else:
            gray = patch
            
        return cv2.Laplacian(gray, cv2.CV_64F).var()
