"""
Setup script for the People Counter application.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.config.settings import MODELS_DIR, LOGS_DIR

def setup_directories():
    """Create necessary directories"""
    print("Creating required directories...")
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    print("Directories created.")

def download_models():
    """Download required models"""
    print("Downloading YOLOv8 model...")
    try:
        from ultralytics import YOLO
        model_path = MODELS_DIR / 'yolov8n.pt'
        if not model_path.exists():
            YOLO('yolov8n.pt').save(str(model_path))
            print(f"Model downloaded to {model_path}")
        else:
            print(f"Model already exists at {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")

def setup_environment():
    """Set up environment variables"""
    print("Creating .env file if it doesn't exist...")
    env_example = project_root / '.env.example'
    env_file = project_root / '.env'
    
    if not env_file.exists() and env_example.exists():
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print(".env file created from example")
    elif not env_file.exists():
        print("Creating basic .env file")
        with open(env_file, 'w') as f:
            f.write("# Camera URL (RTSP, HTTP, or device number for webcams)\n")
            f.write("CAMERA_URL=0\n\n")
            f.write("# Detection confidence threshold (0.0-1.0)\n")
            f.write("CONFIDENCE_THRESHOLD=0.25\n\n")
            f.write("# Flask web server port\n")
            f.write("FLASK_PORT=8888\n")

if __name__ == "__main__":
    print("Setting up People Counter application...")
    setup_directories()
    download_models()
    setup_environment()
    print("Setup complete!")
