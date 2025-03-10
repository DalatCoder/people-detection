"""
Configuration settings for the People Counter application.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

def load_config():
    """Load configuration from environment variables with defaults"""
    config = {
        # Model settings
        'MODEL_PATH': os.getenv('MODEL_PATH', str(MODELS_DIR / 'yolov8n.pt')),
        'CONF_THRESHOLD': float(os.getenv('CONFIDENCE_THRESHOLD', '0.25')),
        
        # Camera settings
        'CAMERA_URL': os.getenv('CAMERA_URL', '0'),  # Default to first webcam
        
        # Web settings
        'WEB_PORT': int(os.getenv('FLASK_PORT', '8888')),
        
        # Performance settings
        'FORCE_CPU': os.getenv('FORCE_CPU', '').lower() in ('true', '1', 'yes'),
    }
    return config

# Model configuration dictionary for compatibility with existing code
MODEL_CONFIG = {
    'model_path': os.getenv('MODEL_PATH', str(MODELS_DIR / 'yolov8n.pt')),
    'conf_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.25'))
}
