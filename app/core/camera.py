"""
Camera utilities for handling different camera types and protocols.
"""
import cv2
import os
from app.config.settings import load_config

def create_camera_capture(source=None):
    """
    Create a camera capture using the appropriate protocol
    
    Args:
        source: Camera source (URL, device index, etc.)
               If None, will use CAMERA_URL from config
    
    Returns:
        cv2.VideoCapture object
    """
    if source is None:
        config = load_config()
        source = config['CAMERA_URL']
    
    # Convert numeric string to int for webcam index
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Auto-detect protocol
    protocol = None
    if isinstance(source, int):
        protocol = 'usb'
    elif isinstance(source, str):
        if source.startswith('rtsp://'):
            protocol = 'rtsp'
        elif '.mjpg' in source or '.mjpeg' in source:
            protocol = 'mjpeg'
        elif '.m3u8' in source:
            protocol = 'hls'
        else:
            protocol = 'http'
    
    # Create capture based on protocol
    stream = cv2.VideoCapture(source)
    
    # Apply protocol-specific settings
    if protocol == 'rtsp':
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        stream.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TRANSPORT_TCP)
    elif protocol == 'usb':
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        stream.set(cv2.CAP_PROP_FPS, 30)
    
    if not stream.isOpened():
        print(f"Failed to open camera source: {source}")
        
    return stream

def release_camera(stream):
    """Safely release camera resources"""
    if stream and stream.isOpened():
        stream.release()
