
# Camera configuration
CAMERA_CONFIG = {
    'ip': '',
    'username': '',
    'password': '',
    'rtsp_url': f'rtsp://[username]:[password]@[ip]:554/stream1',  # Adjust port and stream path if needed
}

# YOLOv8 configuration
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',  # Default smallest YOLOv8 model
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
}
