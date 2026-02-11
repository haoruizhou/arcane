import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class TextureManager:
    """
    Manages the DPG Texture Registry.
    Handles loading images into GPU memory for display.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextureManager, cls).__new__(cls)
            cls._instance._init_registry()
        return cls._instance

    def _init_registry(self):
        self.registry_tag = "texture_registry"
        if not dpg.does_item_exist(self.registry_tag):
            dpg.add_texture_registry(tag=self.registry_tag, show=False)
        self.textures = {} # map path -> texture_tag

    def load_texture(self, image_path: str, image: Image.Image) -> str:
        """
        Loads a PIL Image into the texture registry.
        Returns the texture_tag.
        """
        if image_path in self.textures:
            return self.textures[image_path]

        try:
            # Ensure image is RGBA
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            
            width, height = image.size
            # Convert to normalized float array (0.0 - 1.0) flattened
            data = np.array(image, dtype=np.float32) / 255.0
            data = data.flatten()
            
            tag = f"tex_{image_path}"
            
            # DPG requires unique tags. If reloading, delete old one (though we check cache above)
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
                
            dpg.add_static_texture(width=width, height=height, default_value=data, tag=tag, parent=self.registry_tag)
            self.textures[image_path] = tag
            return tag
            
        except Exception as e:
            logger.error(f"Failed to create texture for {image_path}: {e}")
            return None

    def get_texture(self, image_path: str):
        return self.textures.get(image_path)
