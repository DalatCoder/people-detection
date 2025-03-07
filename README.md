# Camera Object Detection with YOLOv8

This project uses YOLOv8 for real-time object detection using an IP camera.

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv8 model:

   ```bash
   # The script will download it automatically on first run
   # or you can manually download from https://github.com/ultralytics/assets/releases/
   ```

4. Run detection:
   ```bash
   python detect.py
   ```

## Configuration

Camera and model settings can be modified in `config.py`.

## Security Note

The `config.py` file contains camera credentials. In a production environment, consider:

- Using environment variables instead of hardcoded credentials
- Adding `config.py` to `.gitignore` after initial setup
