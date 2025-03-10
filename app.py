from flask import Flask, render_template, Response, jsonify
from detect import PeopleDetector
import time

app = Flask(__name__)

# Initialize the detector
detector = PeopleDetector(use_cpu=False)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/stats')
def stats():
    """Return current detection statistics as JSON"""
    return jsonify(detector.get_stats())

def generate_frames():
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

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start detection in background thread
    detector.start_detection()
    try:
        # Start Flask server
        app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)
    finally:
        # Stop detection when the server exits
        detector.stop_detection()
