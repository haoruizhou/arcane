import rawpy
import numpy as np
from PIL import Image
import threading
import logging
import os

logger = logging.getLogger(__name__)

class ImageLoader:
    """
    Handles loading of RAW images and generation of previews.
    """
    
    @staticmethod
    def load_raw(path: str) -> np.ndarray:
        """
        Load a RAW image and return it as a numpy array (RGB).
        Uses rawpy for high-quality post-processing.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            with rawpy.imread(path) as raw:
                # Postprocess to get a viewable RGB image
                # use_camera_wb=True used to apply camera white balance
                rgb = raw.postprocess(use_camera_wb=True)
                return rgb
        except Exception as e:
            logger.error(f"Failed to load RAW image {path}: {e}")
            raise

    @staticmethod
    def load_preview(path: str, max_size: tuple[int, int] = (1024, 1024)) -> Image.Image:
        """
        Extract a preview from an image file for fast display.
        Supports RAW files (via rawpy embedded preview) and standard formats (JPEG, PNG, etc.).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        # Try RAW processing first
        try:
            with rawpy.imread(path) as raw:
                try:
                    thumb = raw.extract_thumb()
                except rawpy.LibRawError:
                    thumb = None
                
                if thumb and thumb.format == rawpy.ThumbFormat.JPEG:
                    import io
                    img = Image.open(io.BytesIO(thumb.data))
                elif thumb and thumb.format == rawpy.ThumbFormat.BITMAP:
                    img = Image.fromarray(thumb.data)
                else:
                    rgb = raw.postprocess(half_size=True)
                    img = Image.fromarray(rgb)
                
                img.thumbnail(max_size)
                return img
        except rawpy.LibRawFileUnsupportedError:
            # Not a RAW file — fall through to PIL
            pass
        except Exception as e:
            logger.error(f"Failed to load RAW preview for {path}: {e}")
            raise

        # Fallback: standard image formats (JPEG, PNG, TIFF, etc.)
        try:
            img = Image.open(path)
            img = img.convert("RGB")
            img.thumbnail(max_size)
            return img
        except Exception as e:
            logger.error(f"Failed to load preview for {path}: {e}")
            raise
