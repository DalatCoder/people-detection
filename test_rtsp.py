import argparse
import cv2
import time
from rtsp_utils import generate_rtsp_url, test_rtsp_connections

def show_camera_stream(url, display_time=30):
    """Display camera stream for a specified duration to test connection"""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {url}")
        return False
    
    print(f"Successfully connected to: {url}")
    print(f"Displaying stream for {display_time} seconds...")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < display_time:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
        
        frame_count += 1
        cv2.putText(frame, f"Frames: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("RTSP Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    avg_fps = frame_count / max(1, (time.time() - start_time))
    print(f"Average FPS: {avg_fps:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RTSP connections to IP camera')
    parser.add_argument('--ip', type=str, required=True, help='Camera IP address')
    parser.add_argument('--username', type=str, default='admin', help='Camera username')
    parser.add_argument('--password', type=str, required=True, help='Camera password')
    parser.add_argument('--port', type=int, default=554, help='RTSP port')
    parser.add_argument('--auto', action='store_true', help='Auto-test all URL patterns')
    parser.add_argument('--url', type=str, help='Custom RTSP URL to test')
    
    args = parser.parse_args()
    
    if args.auto:
        working_urls = test_rtsp_connections(args.ip, args.username, args.password, args.port)
        if working_urls:
            print("\nTesting video display from first working URL...")
            show_camera_stream(working_urls[0])
        else:
            print("No working RTSP URLs found.")
    elif args.url:
        show_camera_stream(args.url)
    else:
        urls = generate_rtsp_url(args.ip, args.username, args.password, args.port)
        print("Generated RTSP URLs to try:")
        for i, url in enumerate(urls, 1):
            print(f"{i}. {url}")

# Usage: python test_rtsp.py --ip 10.10.45.243 --username admin --password 123abc789 --auto
