#!/usr/bin/env python3
"""
Main entry point for the People Counter application.
"""
import argparse
import os
from app.core.detector import PeopleDetector
from app.web.app import create_app
from app.config.settings import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Run people counting on camera stream")
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--no-display', action='store_true', help='Run without display (headless mode)')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--port', type=int, default=8888, help='Port for web interface')
    parser.add_argument('--camera', type=str, help='Camera URL or device ID')
    parser.add_argument('--confidence', type=float, help='Detection confidence threshold')
    return parser.parse_args()

def run_standalone(args):
    """Run the detector in standalone mode with optional display"""
    from app.core.display import run_detection_with_display
    run_detection_with_display(use_cpu=args.cpu, display=not args.no_display)
    
def run_web_interface(args):
    """Run the web interface with detector"""
    config = load_config()
    if args.port:
        config['WEB_PORT'] = args.port
    
    app = create_app(config)
    app.run(host='0.0.0.0', port=config['WEB_PORT'], debug=False, threaded=True)

if __name__ == "__main__":
    args = parse_args()
    
    # Override configs with command line arguments
    if args.camera:
        os.environ['CAMERA_URL'] = args.camera
    if args.confidence:
        os.environ['CONFIDENCE_THRESHOLD'] = str(args.confidence)
    
    if args.web:
        run_web_interface(args)
    else:
        run_standalone(args)
