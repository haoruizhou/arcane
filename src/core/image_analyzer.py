import numpy as np
import logging
from PIL import Image
from src.core.image_loader import ImageLoader
from src.ml.face_detector import FaceDetector
from src.ml.focus_detector import FocusDetector
from src.ml.eye_detector import EyeOpennessDetector
from src.ml.face_recognizer import FaceRecognizer
from src.ml.semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """
    Encapsulates the ML pipeline for analyzing images.
    """
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.semantic_analyzer = SemanticAnalyzer()
        
    def compute_dhash(self, image: Image.Image, hash_size: int = 8) -> str:
        """
        Compute dHash (Difference Hash) for an image.
        Returns a hex string representing the 64-bit hash.
        """
        # Resize to (hash_size + 1, hash_size)
        # Using 9x8 for 64 bits
        resized = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS).convert("L")
        pixels = list(resized.getdata())
        
        # Compare adjacent pixels
        diff = []
        for row in range(hash_size):
            for col in range(hash_size):
                left_pixel = pixels[row * (hash_size + 1) + col]
                right_pixel = pixels[row * (hash_size + 1) + col + 1]
                diff.append(left_pixel > right_pixel)
                
        # Convert binary array to hex string
        decimal_value = 0
        for index, value in enumerate(diff):
            if value:
                decimal_value += 2**index
                
        return hex(decimal_value)[2:]

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
                det['sharpness_score'] = sharpness
                # Remove legacy key if strictly needed, but let's just add new one and keep 'sharpness' for now if unsure
                # actually let's stick to 'sharpness' but know it is 0-100 now
                det['sharpness'] = sharpness
                
                if sharpness > max_focus:
                    max_focus = sharpness
                
                if det.get('landmarks') is not None:
                    eye_score = EyeOpennessDetector.check_eyes(img_np, det['landmarks'])
                    det['eye_openness_score'] = eye_score
                    # Legacy support / Interpretation
                    det['eye_status'] = "Open" if eye_score > 50.0 else "Closed"
                
                # Normalize confidence to 0-100
                if 'confidence' in det:
                     det['confidence_score'] = det['confidence'] * 100.0

                # Extract Face Embedding
                # Crop face with some margin? FaceNet expects tight crop usually? 
                # Let's crop exactly to box for now.
                x1, y1, x2, y2 = map(int, box)
                # Clamp
                h, w = img_np.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    face_crop = Image.fromarray(img_np[y1:y2, x1:x2])
                    face_emb = self.face_recognizer.get_embedding(face_crop)
                    if face_emb is not None:
                        det['embedding'] = face_emb.tolist()

            result["detections"] = detections
            
            if not detections:
                # If no faces, check global sharpness
                global_sharpness = FocusDetector.measure_sharpness(img_np)
                result["focus_score"] = global_sharpness
            else:
                result["focus_score"] = max_focus
            
            # Compute dHash
            result["dhash"] = self.compute_dhash(img)
            
            # Compute Semantic Embedding (CLIP)
            embedding = self.semantic_analyzer.get_embedding(img)
            # Store as list for JSON serialization
            if embedding is not None:
                result["embedding"] = embedding.tolist()
            
            result["overall_score"] = result["focus_score"]  
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            result["error"] = str(e)
            
        return result
