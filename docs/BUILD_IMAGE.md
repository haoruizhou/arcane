  Steps to build the offline app image:


   1. Pre-download all models:
   1     python src/ml/download_models.py
   2. Install PyInstaller:
   1     pip install pyinstaller
   3. Build the application:
   1     pyinstaller arcane.spec