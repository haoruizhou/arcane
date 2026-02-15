import sys
import os

# Fix for OpenMP crash (multiple libraries loading libomp)
# Aggressively disable threading for runtimes that might conflict
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path so we can import 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gui.gui_manager import GuiManager
import logging

import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def main():
    # Register global exception handler
    sys.excepthook = handle_exception
    
    try:
        app = GuiManager()
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error in main loop: {e}", exc_info=True)
        # Optional: Keep window open or print to stderr if GUI is gone
        traceback.print_exc()

if __name__ == "__main__":
    main()
