import json
import os
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AnalysisCache:
    """
    Manages caching of image analysis results to a JSON file.
    """
    CACHE_FILENAME = ".arcane_cache.json"
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.cache_path = os.path.join(folder_path, self.CACHE_FILENAME)
        self.cache = {}
        self.dirty = False
        self.load()
        
    def load(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries.")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def get(self, image_path: str) -> dict:
        """
        Get cached result if valid.
        Returns None if not found or stale.
        """
        filename = os.path.basename(image_path)
        if filename in self.cache:
            entry = self.cache[filename]
            
            # Check modification time
            try:
                mtime = os.path.getmtime(image_path)
                if entry.get('mtime', 0) == mtime:
                    return entry['data']
            except OSError:
                pass
                
        return None

    def set(self, image_path: str, result: dict):
        """Update cache with new result."""
        filename = os.path.basename(image_path)
        try:
            mtime = os.path.getmtime(image_path)
            
            # Deep copy and serialize numpy types
            serializable_result = self._make_serializable(result)
            
            self.cache[filename] = {
                'mtime': mtime,
                'data': serializable_result
            }
            self.dirty = True
        except Exception as e:
            logger.error(f"Error setting cache for {filename}: {e}")

    def save(self):
        """Save cache to disk if modified."""
        if not self.dirty:
            return
            
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)
            self.dirty = False
            logger.info("Cache saved.")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _make_serializable(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() if k != "thumbnail"} # Skip thumbnail object
        elif isinstance(obj, list):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Skip non-serializable objects (PIL Images, etc.)
            return None
