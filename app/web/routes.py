"""
Web routes for the Flask application.
"""
from flask import render_template, Response, jsonify
import time

def register_routes(app, detector):
    """
    Register routes with the Flask application
    
    Args:
        app: Flask application instance
        detector: PeopleDetector instance
    """
    @app.route('/')
    def index():
        """Render the main page"""
        return render_template('index.html')

    @app.route('/stats')
    def stats():
        """Return current detection statistics as JSON"""
        return jsonify(detector.get_stats())

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route"""
        return Response(generate_frames(detector),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
                        
def generate_frames(detector):
    """Generate video frames for streaming"""
    while True:
        # Get the latest frame
        frame = detector.get_latest_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # No frame available yet, short delay
            time.sleep(0.01)
