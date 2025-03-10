import cv2
import numpy as np
import time
import os
import argparse
from ultralytics import YOLO
from camera_utils import create_camera_capture, release_camera
from config import MODEL_CONFIG
from gpu_utils import check_gpu_support, select_device, optimize_performance

def run_detection(use_cpu=False, display=True):
    # Check GPU support and optimize performance
    gpu_info = check_gpu_support()
    optimize_performance(gpu_info)
    device = select_device(gpu_info, force_cpu=use_cpu)
    
    print(f"Running detection on device: {device}")
    
    # Load YOLOv8 model with the selected device
    model = YOLO(MODEL_CONFIG['model_path'])
    if device != 'cpu':
        print(f"Using GPU acceleration with {device}")
        # For YOLOv8, device is set during inference
    else:
        print("Using CPU for inference")
    
    # Connect to camera using improved RTSP handling
    stream = create_camera_capture()
    
    # Track FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    frames_processed = 0
    
    # Track person count
    max_people_count = 0
    
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
            results = model(frame, conf=MODEL_CONFIG['conf_threshold'], device=device)
            frames_processed += 1
            
            # Count people in the frame (class 0 is 'person' in COCO dataset)
            people_count = sum(1 for box in results[0].boxes if box.cls == 0)
            
            # Update maximum people count
            max_people_count = max(max_people_count, people_count)
            
            # Log performance metrics every 100 frames
            if frames_processed % 100 == 0:
                print(f"Processed {frames_processed} frames. Current FPS: {fps:.2f}, People: {people_count}")
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Add FPS and people count info
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"People: {people_count}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Max People: {max_people_count}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if display:
                # Display the annotated frame
                cv2.imshow("People Counter", annotated_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    finally:
        # Release resources
        release_camera(stream)
        if display:
            cv2.destroyAllWindows()
        print(f"Processed {frames_processed} frames in total.")
        print(f"Maximum number of people detected: {max_people_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run people counting on camera stream")
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--no-display', action='store_true', help='Run without display (headless mode)')
    args = parser.parse_args()
    
    run_detection(use_cpu=args.cpu, display=not args.no_display)
