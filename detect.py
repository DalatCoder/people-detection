import cv2
import numpy as np
import time
import os
import argparse
import threading
from ultralytics import YOLO
from camera_utils import create_camera_capture, release_camera
from config import MODEL_CONFIG
from gpu_utils import check_gpu_support, select_device, optimize_performance

class PeopleDetector:
    def __init__(self, use_cpu=False):
        # Check GPU support and optimize performance
        self.gpu_info = check_gpu_support()
        optimize_performance(self.gpu_info)
        self.device = select_device(self.gpu_info, force_cpu=use_cpu)
        
        print(f"Running detection on device: {self.device}")
        
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
        stream = create_camera_capture()
        
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


def run_detection(use_cpu=False, display=True):
    detector = PeopleDetector(use_cpu=use_cpu)
    detector.start_detection()
    
    if display:
        try:
            while True:
                frame = detector.get_latest_frame()
                if frame is not None:
                    # Convert bytes back to frame for display
                    nparr = np.frombuffer(frame, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    cv2.imshow("People Counter", img)
                    
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.01)  # Short delay if no frame is available
        finally:
            detector.stop_detection()
            cv2.destroyAllWindows()
    else:
        try:
            # Just run in the background and print stats periodically
            while True:
                time.sleep(10)  # Print stats every 10 seconds
                stats = detector.get_stats()
                print(f"Current stats: FPS={stats['fps']:.2f}, People={stats['people_count']}, Max People={stats['max_people_count']}")
        except KeyboardInterrupt:
            detector.stop_detection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run people counting on camera stream")
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--no-display', action='store_true', help='Run without display (headless mode)')
    args = parser.parse_args()
    
    run_detection(use_cpu=args.cpu, display=not args.no_display)
