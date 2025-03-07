import cv2
from config import CAMERA_CONFIG
from rtsp_utils import RTSPVideoStream, test_rtsp_connections

def find_working_rtsp_url():
    """Find a working RTSP URL for the camera"""
    ip = CAMERA_CONFIG['ip']
    username = CAMERA_CONFIG['username']
    password = CAMERA_CONFIG['password']
    
    # Try the configured RTSP URL first
    configured_url = CAMERA_CONFIG.get('rtsp_url')
    if configured_url:
        cap = cv2.VideoCapture(configured_url)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Using configured RTSP URL: {configured_url}")
                return configured_url
    
    # Test different RTSP URL patterns
    print("Testing different RTSP URL patterns...")
    working_urls = test_rtsp_connections(ip, username, password)
    
    if working_urls:
        print(f"Found {len(working_urls)} working RTSP URLs. Using the first one.")
        return working_urls[0]
    else:
        # Fallback to HTTP
        http_url = f"http://{ip}/video"
        print(f"No working RTSP URLs found. Trying HTTP: {http_url}")
        cap = cv2.VideoCapture(http_url)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return http_url
    
    raise ConnectionError(f"Could not connect to camera at {ip}")

def create_camera_capture():
    """Create and return a camera capture object."""
    # Find a working URL
    rtsp_url = find_working_rtsp_url()
    print(f"Connecting to camera at {rtsp_url}")
    
    # Use threaded RTSP stream for better performance
    stream = RTSPVideoStream(rtsp_url).start()
    return stream

def release_camera(cap):
    """Safely release the camera resource."""
    if cap is not None:
        if hasattr(cap, 'stop'):  # RTSPVideoStream instance
            cap.stop()
        else:  # cv2.VideoCapture instance
            cap.release()
