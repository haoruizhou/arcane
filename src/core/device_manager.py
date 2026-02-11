import torch
import logging

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Manages the selection of the computation device (CPU, CUDA, MPS).
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._init_device()
        return cls._instance

    def _init_device(self):
        """Initialize the best available device."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {self.device_name}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = "mps"
            self.device_name = "Apple Metal Performance Shaders"
            logger.info("Using MPS (Metal) device")
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
            self.device_name = "CPU"
            logger.info("Using CPU device")

    def get_device(self) -> torch.device:
        """Return the torch device."""
        return self.device

    def get_device_info(self) -> dict:
        """Return information about the current device."""
        return {
            "type": self.device_type,
            "name": self.device_name,
            "torch_version": torch.__version__
        }
