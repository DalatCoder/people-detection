"""
Display utilities for showing detection results.
"""
import cv2
import numpy as np
import time
from app.core.detector import PeopleDetector
from app.config.settings import load_config

def run_detection_with_display(use_cpu=False, display=True):
    """Run detection with optional display window"""
    config = load_config()
    
    detector = PeopleDetector(
        use_cpu=use_cpu or config['FORCE_CPU'], 
        camera_url=config['CAMERA_URL']
    )
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
