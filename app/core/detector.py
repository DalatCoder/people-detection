"""
Core detector module for people counting.
"""
import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO

from app.config.settings import MODEL_CONFIG
from app.core.camera import create_camera_capture, release_camera
from app.core.gpu import check_gpu_support, select_device, optimize_performance

class PeopleDetector:
    def __init__(self, use_cpu=False, camera_url=None):
        # Check GPU support and optimize performance
        self.gpu_info = check_gpu_support()
        optimize_performance(self.gpu_info)
        self.device = select_device(self.gpu_info, force_cpu=use_cpu)
        
        print(f"Running detection on device: {self.device}")
        
        # Store camera URL if provided
        self.camera_url = camera_url
        
        # Load YOLOv8 model with the selected device
        self.model = YOLO(MODEL_CONFIG['model_path'])
        if self.device != 'cpu':
            print(f"Using GPU acceleration with {self.device}")
        else:
            print("Using CPU for inference")
        
        # Stats tracking
        self.fps = 0
        self.frames_processed = 0
        self.people_count = 0
        self.max_people_count = 0
        self.latest_frame = None
        
        # Thread control
        self.running = False
        self.detection_thread = None
    
    def start_detection(self):
        """Start detection in a separate thread"""
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            return True
        return False
    
    def stop_detection(self):
        """Stop the detection thread"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5.0)
        return not (self.detection_thread and self.detection_thread.is_alive())
    
    def get_stats(self):
        """Return current detection statistics"""
        return {
            'fps': self.fps,
            'frames_processed': self.frames_processed,
            'people_count': self.people_count,
            'max_people_count': self.max_people_count
        }
    
    def get_latest_frame(self):
        """Return the latest processed frame with annotations"""
        return self.latest_frame
    
    def _detection_loop(self):
        """Main detection loop running in a thread"""
        # Connect to camera
        stream = create_camera_capture(self.camera_url)
        
        # Track FPS
        fps_start_time = time.time()
        fps_counter = 0
        
        try:
            while self.running:
                success, frame = stream.read()
                
                if not success:
                    print("Failed to receive frame from camera.")
                    time.sleep(0.1)  # Small delay before retrying
                    continue
                    
                # Calculate FPS
                fps_counter += 1
                if (time.time() - fps_start_time) > 1.0:
                    self.fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Run YOLOv8 inference on the frame
                results = self.model(frame, conf=MODEL_CONFIG['conf_threshold'], device=self.device)
                self.frames_processed += 1
                
                # Count people in the frame (class 0 is 'person' in COCO dataset)
                self.people_count = sum(1 for box in results[0].boxes if box.cls == 0)
                
                # Update maximum people count
                self.max_people_count = max(self.max_people_count, self.people_count)
                
                # Log performance metrics every 100 frames
                if self.frames_processed % 100 == 0:
                    print(f"Processed {self.frames_processed} frames. Current FPS: {self.fps:.2f}, People: {self.people_count}")
                
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                
                # Add FPS and people count info
                cv2.putText(annotated_frame, f"FPS: {self.fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"People: {self.people_count}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Max People: {self.max_people_count}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Store the latest frame
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                self.latest_frame = buffer.tobytes()
                    
        finally:
            # Release resources
            release_camera(stream)
            print(f"Detection stopped. Processed {self.frames_processed} frames in total.")
            print(f"Maximum number of people detected: {self.max_people_count}")
