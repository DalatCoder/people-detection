import cv2
import numpy as np
import time
from ultralytics import YOLO
from camera_utils import create_camera_capture, release_camera
from config import MODEL_CONFIG

def run_detection():
    # Load YOLOv8 model
    model = YOLO(MODEL_CONFIG['model_path'])
    
    # Connect to camera using improved RTSP handling
    stream = create_camera_capture()
    
    # Track FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            success, frame = stream.read()
            
            if not success:
                print("Failed to receive frame from camera.")
                time.sleep(0.1)  # Small delay before retrying
                continue
                
            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=MODEL_CONFIG['conf_threshold'])
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Add FPS info
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the annotated frame
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release resources
        release_camera(stream)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()
