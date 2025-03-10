# Camera configuration
CAMERA_CONFIG = {
    # --- IP Camera Configuration ---
    'ip': '192.168.1.100',            # IP address of your camera
    'username': 'username',           # Camera login username
    'password': 'password',           # Camera login password
    
    # RTSP URL examples (uncomment and modify one that works with your camera)
    'rtsp_url': f'rtsp://username:password@192.168.1.100:554/stream1',
    # 'rtsp_url': f'rtsp://username:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0',
    
    # Connection settings
    'connection_timeout': 30,         # Seconds to wait before timeout
    'reconnect_attempts': 5,          # Number of reconnection attempts
    'reconnect_delay': 3,             # Seconds to wait between reconnection attempts
    'fallback_to_http': True,         # Try HTTP if RTSP fails
    
    # Alternative RTSP paths to try if the main URL fails
    'alt_rtsp_paths': [
        'Streaming/Channels/101',
        'live/ch0',
        'cam/realmonitor?channel=1&subtype=0',
        'h264Preview_01_main',
        'h264Preview_01_sub',
        'live0',
        'live/mpeg4',
        'video1',
    ],
    
    # --- Webcam Configuration ---
    'use_webcam': False,              # Set to True to use laptop's webcam instead of IP camera
    'webcam_id': 0                    # Device ID for webcam (0=default/built-in camera, 1=external)
}

# YOLOv8 configuration
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',       # Path to YOLOv8 model file (n=nano, s=small, m=medium, l=large, x=xlarge)
    'conf_threshold': 0.25,           # Detection confidence threshold (0.0-1.0)
    'iou_threshold': 0.45,            # Intersection over Union threshold for NMS
}

# Web application configuration
WEB_CONFIG = {
    'host': '0.0.0.0',                # 0.0.0.0 allows external connections, 127.0.0.1 for local only
    'port': 8888,                     # Web server port
    'debug': True,                    # Enable Flask debug mode (disable in production)
    'max_detection_fps': 5,           # Maximum FPS for detection processing (lower = less CPU)
    'max_streaming_fps': 20           # Maximum FPS for video streaming (lower = less bandwidth)
}
