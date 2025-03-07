import cv2
import time
import threading
from queue import Queue

class RTSPVideoStream:
    """
    Class to handle RTSP video streaming with improved reliability
    and performance using threading.
    """
    def __init__(self, rtsp_url, queue_size=128):
        self.rtsp_url = rtsp_url
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.reconnect_timeout = 5  # seconds to wait before reconnection attempt
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.frame_counter = 0
        self.last_frame = None
        
    def start(self):
        """Start the thread to read frames from the video stream"""
        self.stopped = False
        threading.Thread(target=self._update, args=()).start()
        return self
        
    def _update(self):
        """Update frames in a separate thread"""
        while not self.stopped:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                print(f"Failed to connect after {self.max_reconnect_attempts} attempts. Stopping.")
                self.stopped = True
                break
                
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    raise ConnectionError(f"Could not connect to RTSP stream at {self.rtsp_url}")
                
                self.reconnect_attempts = 0  # Reset counter on successful connection
                print(f"Successfully connected to RTSP stream: {self.rtsp_url}")
                
                # Loop to continuously read frames
                while not self.stopped and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame, attempting to reconnect...")
                        break
                        
                    # Clear the queue if it's getting too full to avoid lag
                    if self.queue.qsize() > 10:
                        try:
                            while self.queue.qsize() > 5:
                                self.queue.get_nowait()
                        except:
                            pass
                    
                    # Add frame to queue
                    self.queue.put(frame)
                    self.last_frame = frame
                    self.frame_counter += 1
                    
                # Release resources before reconnecting
                cap.release()
                
            except Exception as e:
                print(f"Error in RTSP stream: {e}")
                
            # Wait before reconnection attempt
            self.reconnect_attempts += 1
            print(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {self.reconnect_timeout} seconds...")
            time.sleep(self.reconnect_timeout)
    
    def read(self):
        """Return the most recent frame"""
        if self.queue.empty():
            return False, self.last_frame  # Return the last available frame if queue is empty
        else:
            return True, self.queue.get()
    
    def stop(self):
        """Stop the thread and release resources"""
        self.stopped = True

def generate_rtsp_url(ip, username, password, port=554, path="stream1"):
    """Generate common RTSP URL formats to try"""
    # Common RTSP URL patterns for different camera manufacturers
    rtsp_patterns = [
        f"rtsp://{username}:{password}@{ip}:{port}/{path}",                          # Generic
        f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/1",            # Hikvision
        f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0", # Dahua
        f"rtsp://{username}:{password}@{ip}:{port}/axis-media/media.amp",            # Axis
        f"rtsp://{username}:{password}@{ip}:{port}/h264Preview_01_main",             # Amcrest
        f"rtsp://{username}:{password}@{ip}:{port}/live/ch00_0",                     # Reolink
    ]
    
    return rtsp_patterns

def test_rtsp_connections(ip, username, password, port=554):
    """Test multiple RTSP URL patterns and return working ones"""
    rtsp_urls = generate_rtsp_url(ip, username, password, port)
    working_urls = []
    
    for url in rtsp_urls:
        print(f"Testing connection to: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Successfully connected to: {url}")
                working_urls.append(url)
            else:
                print(f"✗ Connected but couldn't read frame from: {url}")
        else:
            print(f"✗ Failed to connect to: {url}")
        cap.release()
    
    return working_urls
