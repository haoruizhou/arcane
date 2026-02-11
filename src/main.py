import sys
import os

# Add project root to sys.path so we can import 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gui.gui_manager import GuiManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    app = GuiManager()
    app.run()

if __name__ == "__main__":
    main()
