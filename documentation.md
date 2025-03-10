# People Counter System - Technical Documentation

This document provides detailed technical information about the People Counter system, including environment setup, camera integration, YOLOv8 implementation, and the client-server architecture.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Camera Integration](#camera-integration)
- [People Detection with YOLOv8](#people-detection-with-yolov8)
- [Client-Server Architecture](#client-server-architecture)

## Environment Setup

### System Requirements

#### Hardware

- **Processor**: Multi-core CPU (Intel i5/i7 or AMD equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
  - At least 4GB VRAM for optimal performance
- **Storage**: 500MB for application and model

#### Software

- **Operating System**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: Version 3.7 to 3.10
- **CUDA**: Version 10.2+ (for GPU acceleration)
- **cuDNN**: Version compatible with installed CUDA

### Installation Steps

1. **Clone Repository**

   ```bash
   git clone https://github.com/yourusername/camera-detection.git
   cd camera-detection
   ```

2. **Create Virtual Environment (recommended)**

   ```bash
   # Using venv
   python -m venv venv

   # Activate on Windows
   venv\Scripts\activate

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 Model**

   The system uses YOLOv8 models from Ultralytics. The default configuration expects the model at a predefined path specified in `config.py`. You can download the pre-trained model using:

   ```bash
   # Inside your Python environment
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').download()"
   ```

   Update the `config.py` file to point to your downloaded model:

   ```python
   MODEL_CONFIG = {
       'model_path': 'path/to/yolov8n.pt',  # Update this path
       'conf_threshold': 0.25
   }
   ```

5. **Configure Camera Settings**

   Edit `camera_utils.py` to set up your camera connection parameters:

   ```python
   def create_camera_capture():
       # For RTSP stream
       stream_url = "rtsp://username:password@camera-ip:554/stream"

       # For webcam
       # stream_url = 0  # Use 0 for default webcam, 1 for external webcam

       return cv2.VideoCapture(stream_url)
   ```

6. **Test Installation**

   Run a basic test to verify everything is working:

   ```bash
   python detect.py --cpu  # Test with CPU
   ```

## Camera Integration

### Camera Connection Methods

The system supports multiple camera connection types:

1. **Local Webcams**: Connected directly to the system.
2. **IP Cameras**: Connected over the network using RTSP streams.
3. **Video Files**: For testing with recorded footage.

### Implementation in `camera_utils.py`

The camera handling is abstracted in the `camera_utils.py` file, providing functions for:

- **Creating camera connections**: Establishing and configuring stream parameters
- **Handling disconnections**: Automatic retry mechanisms
- **Releasing resources**: Proper cleanup when shutting down

#### How Real-time Data is Handled

1. **Stream Initialization**:

   ```python
   def create_camera_capture():
       # Example for RTSP stream with optimization
       stream = cv2.VideoCapture(CAMERA_URL)

       # Configure buffer size to reduce latency
       stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

       # Set resolution (optional)
       stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
       stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

       return stream
   ```

2. **Frame Reading Process**:

   - The system continuously reads frames in a separate thread
   - This non-blocking approach prevents the UI from freezing
   - Includes timeout handling for unresponsive cameras

3. **Error Handling**:
   - Automatic reconnection attempts when frames cannot be read
   - Logging of connection issues
   - Graceful degradation when camera feed is temporarily unavailable

### Best Practices for Camera Integration

1. **Buffer Management**: Minimize buffer size to reduce latency
2. **Resolution Control**: Balance between quality and performance
3. **Thread Safety**: Ensure proper synchronization when accessing camera from multiple threads
4. **Timeouts**: Implement read timeouts to prevent blocking on failed connections
5. **Reconnection Strategy**: Use exponential backoff for reconnection attempts

### Camera Connection Protocols

#### RTSP Protocol

**Real Time Streaming Protocol (RTSP)** is the most common protocol used for IP cameras and offers several advantages:

1. **Overview**:

   - A network control protocol designed for use in entertainment and communications systems
   - Operates on port 554 by default
   - Establishes and controls media sessions between endpoints

2. **How RTSP Works**:

   - Control mechanism: Client sends commands (DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN)
   - Transport protocol: Usually RTP (Real-time Transport Protocol) over UDP or TCP
   - Session management: Maintains session state between client and server

3. **RTSP URL Structure**:

   ```
   rtsp://[username:password@]ip_address[:port]/stream_path
   ```

   Example:

   ```
   rtsp://admin:password123@192.168.1.100:554/stream1
   ```

4. **Implementation in Our System**:

   ```python
   def create_rtsp_capture(ip, username=None, password=None, port=554, path="stream1"):
       auth = f"{username}:{password}@" if username and password else ""
       url = f"rtsp://{auth}{ip}:{port}/{path}"

       # OpenCV RTSP optimization
       stream = cv2.VideoCapture(url)
       stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

       # Use TCP instead of UDP (more reliable but slightly higher latency)
       stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
       stream.set(cv2.CAP_PROP_RTSP_TRANSPORT, 'tcp')  # Use 'udp' for lower latency

       return stream
   ```

5. **Advantages**:

   - Widespread support among IP cameras
   - Low latency streaming
   - Bidirectional control (can send commands to camera)
   - Efficient bandwidth usage

6. **Disadvantages**:
   - May be blocked by firewalls
   - Authentication methods can be limited
   - Requires specific port openings in network infrastructure

#### MJPEG Protocol

**Motion JPEG (MJPEG)** is another common protocol, especially for web-based camera viewing:

1. **Overview**:

   - Consists of a sequence of individual JPEG images
   - Each frame is compressed separately as a JPEG image
   - Typically delivered over HTTP

2. **Implementation**:

   ```python
   def create_mjpeg_capture(url):
       stream = cv2.VideoCapture(url)
       return stream
   ```

   Example URL:

   ```
   http://192.168.1.100/video.mjpg
   ```

3. **Advantages**:

   - Works through most firewalls (uses HTTP, typically port 80)
   - Simple implementation
   - Better frame-by-frame quality than H.264 at similar bitrates

4. **Disadvantages**:
   - Higher bandwidth usage than H.264/H.265
   - Higher latency than RTSP
   - No standardized control protocol

#### HTTP/HTTPS Streams

Many modern cameras support direct HTTP or HTTPS streaming:

1. **Overview**:

   - Camera provides video stream via web server
   - Can use various formats (MJPEG, HLS, DASH)
   - Usually requires specific URL paths

2. **HLS (HTTP Live Streaming)**:

   - Developed by Apple
   - Uses .m3u8 playlist files and .ts transport stream segments
   - Example URL: `https://192.168.1.100/stream/playlist.m3u8`

3. **DASH (Dynamic Adaptive Streaming over HTTP)**:

   - Open standard
   - Uses MPD (Media Presentation Description) and segments
   - Adaptive bitrate streaming

4. **Advantages**:

   - Works through firewalls
   - Supports standard HTTP authentication
   - Can use SSL/TLS encryption

5. **Disadvantages**:
   - Higher latency (typically 5-30 seconds)
   - More complex to implement adaptive streaming

#### USB/DirectShow Cameras

Local cameras connected directly to the system:

1. **Implementation in OpenCV**:

   ```python
   def create_local_camera_capture(camera_index=0):
       # camera_index: 0 for default webcam, 1+ for additional cameras
       stream = cv2.VideoCapture(camera_index)

       # Set resolution and frame rate
       stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
       stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
       stream.set(cv2.CAP_PROP_FPS, 30)

       return stream
   ```

2. **Advantages**:

   - Lowest latency option
   - No network configuration required
   - Higher reliability than network cameras

3. **Disadvantages**:
   - Limited by physical USB connections
   - Cable length limitations
   - May require device-specific drivers

### Protocol Selection Guide

| Protocol | Latency | Firewall Friendly | Bandwidth Usage | Setup Complexity | Best For                                     |
| -------- | ------- | ----------------- | --------------- | ---------------- | -------------------------------------------- |
| RTSP     | Low     | No                | Low             | Medium           | Security systems, low-latency monitoring     |
| MJPEG    | Medium  | Yes               | High            | Low              | Simple web integration, quality requirements |
| HLS/DASH | High    | Yes               | Low             | High             | Public broadcasts, mobile viewers            |
| USB      | Lowest  | N/A               | N/A             | Low              | Local monitoring, development testing        |

### Handling Multiple Camera Protocols

Our system can be extended to support multiple protocols with a unified interface:

```python
def create_camera_capture(source, protocol=None):
    """
    Create a camera capture using the appropriate protocol

    Args:
        source: Camera source (URL, device index, etc.)
        protocol: Protocol to use ('rtsp', 'mjpeg', 'http', 'usb')
                  If None, will try to detect from source

    Returns:
        cv2.VideoCapture object
    """
    # Auto-detect protocol if not specified
    if protocol is None:
        if isinstance(source, int):
            protocol = 'usb'
        elif source.startswith('rtsp://'):
            protocol = 'rtsp'
        elif '.mjpg' in source or '.mjpeg' in source:
            protocol = 'mjpeg'
        elif '.m3u8' in source:
            protocol = 'hls'
        else:
            protocol = 'http'

    # Create capture based on protocol
    if protocol == 'rtsp':
        stream = cv2.VideoCapture(source)
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        stream.set(cv2.CAP_PROP_RTSP_TRANSPORT, 'tcp')
    elif protocol == 'usb':
        stream = cv2.VideoCapture(int(source) if isinstance(source, str) else source)
    else:  # mjpeg, http, hls
        stream = cv2.VideoCapture(source)

    return stream
```

## People Detection with YOLOv8

### YOLOv8 Overview

YOLO (You Only Look Once) is a state-of-the-art object detection algorithm. The system uses YOLOv8, the latest version from Ultralytics, which offers improvements in:

- Detection accuracy
- Processing speed
- Multi-object tracking capabilities
- Instance segmentation

### Implementation in `detect.py`

The `PeopleDetector` class handles all aspects of detection:

#### Model Loading

```python
# Model initialization with device selection
self.model = YOLO(MODEL_CONFIG['model_path'])
```

#### Detection Process

The detection loop runs continuously in a separate thread:

1. **Frame Acquisition**: Get frame from camera stream
2. **Preprocessing**: Convert to format expected by YOLOv8
3. **Inference**: Run detection on the image
   ```python
   results = self.model(frame, conf=MODEL_CONFIG['conf_threshold'], device=self.device)
   ```
4. **Post-processing**: Filter results to extract people detections
   ```python
   self.people_count = sum(1 for box in results[0].boxes if box.cls == 0)
   ```
5. **Visualization**: Annotate frame with bounding boxes and counts
6. **Statistics Update**: Update count metrics and FPS calculation

#### People Counting Logic

The system counts people by:

1. Filtering YOLOv8 detection results for class ID 0 (person)
2. Counting the number of bounding boxes with sufficient confidence
3. Tracking maximum counts over time
4. Displaying the counts on the annotated frame and web interface

#### Performance Optimization

1. **GPU Acceleration**:

   - Automatic detection of available CUDA devices
   - Model loading on GPU when available
   - Optimized tensor operations

2. **Inference Optimization**:

   - Batch processing when applicable
   - Half-precision (FP16) inference when supported
   - Appropriate confidence thresholds to reduce false positives

3. **Threading Model**:
   - Detection runs in a background thread
   - Non-blocking frame acquisition
   - Thread-safe communication between detector and web server

### Advanced Detection Features

1. **Confidence Threshold Tuning**:

   - Adjust `conf_threshold` in `config.py` to balance between detection sensitivity and false positives

2. **Model Selection**:

   - YOLOv8 comes in different sizes (nano, small, medium, large)
   - Smaller models are faster but less accurate
   - Larger models are more accurate but require more computational resources

3. **Custom Training** (for specialized environments):
   - The system can be extended to use custom-trained models
   - Useful for specific environments or when standard person detection is insufficient

## Client-Server Architecture

### Server-side Implementation

The server component is built with Flask and handles:

- Video processing
- Detection logic
- API endpoints
- Real-time statistics

#### Core Components

1. **Flask Application (`app.py`)**:

   - Initializes web server and routes
   - Manages detector lifecycle
   - Provides API endpoints for stats and video stream

2. **Video Streaming Implementation**:

   ```python
   @app.route('/video_feed')
   def video_feed():
       return Response(generate_frames(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')

   def generate_frames():
       while True:
           frame = detector.get_latest_frame()
           if frame is not None:
               yield (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
           else:
               time.sleep(0.01)  # Short delay if no frame is available
   ```

3. **Statistics API**:
   ```python
   @app.route('/stats')
   def stats():
       return jsonify(detector.get_stats())
   ```

### Client-side Implementation

The client component is a web interface that:

- Displays the video stream
- Shows real-time statistics
- Updates dynamically without page refreshes

#### Key Technologies

1. **HTML/CSS** (`index.html`):

   - Responsive layout
   - Statistics display boxes
   - Video container

2. **JavaScript**:

   - Periodic AJAX calls to fetch updated statistics
   - Dynamic DOM updates without page reloads

   ```javascript
   setInterval(function () {
     fetch("/stats")
       .then((response) => response.json())
       .then((data) => {
         document.getElementById("current-count").textContent =
           data.people_count;
         document.getElementById("max-count").textContent =
           data.max_people_count;
         document.getElementById("fps").textContent = data.fps.toFixed(1);
         document.getElementById("frames").textContent = data.frames_processed;
       });
   }, 1000);
   ```

3. **Video Streaming**:
   - Uses multipart HTTP response for streaming
   - Browser-native image rendering
   ```html
   <img src="{{ url_for('video_feed') }}" alt="Video stream" />
   ```

### Data Flow

1. **Camera → Detector**:

   - Camera frames are continuously captured
   - Frames are processed by YOLOv8 for detection

2. **Detector → Flask Server**:

   - Processed frames are encoded as JPEG
   - Detection statistics are stored in memory

3. **Flask Server → Browser**:

   - Video frames are streamed as multipart HTTP response
   - Statistics are served as JSON via API endpoints

4. **Browser → User**:
   - Video is displayed using standard HTML elements
   - Statistics are updated using JavaScript

### Scaling Considerations

1. **Multiple Camera Support**:

   - Create detector instances for each camera
   - Implement camera selection in the UI
   - Consider resource allocation per camera

2. **Performance Optimization**:

   - Reduce frame resolution for transmission
   - Implement adaptive frame rate based on network conditions
   - Consider using WebRTC for lower latency in production

3. **Production Deployment**:
   - Use Gunicorn or uWSGI instead of Flask's development server
   - Configure Nginx as a reverse proxy
   - Implement proper authentication for secure access

## Troubleshooting

### Common Issues and Solutions

1. **Camera Connection Failures**:

   - Verify network connectivity to IP cameras
   - Check RTSP URL format and authentication
   - Ensure proper permissions for accessing webcam devices

2. **GPU Detection Issues**:

   - Verify CUDA installation with `nvidia-smi`
   - Ensure compatible versions of CUDA, cuDNN and PyTorch
   - Check GPU memory availability

3. **Performance Problems**:

   - Reduce resolution if FPS is too low
   - Try a smaller YOLOv8 model variant
   - Check for background processes consuming resources

4. **Web Interface Issues**:
   - Clear browser cache if statistics aren't updating
   - Check browser console for JavaScript errors
   - Verify network connectivity between client and server

## Advanced Configuration

### Environment Variables

The system supports configuration through environment variables:

- `CAMERA_URL`: Override the default camera URL
- `MODEL_PATH`: Specify an alternate YOLOv8 model
- `CONFIDENCE_THRESHOLD`: Set detection confidence threshold
- `FLASK_PORT`: Change the web server port

### Command Line Options

Advanced options available through command line:

```bash
# Run with specific camera URL
python app.py --camera rtsp://192.168.1.100/stream

# Set specific GPU device
python app.py --gpu 0

# Adjust confidence threshold
python app.py --confidence 0.4

# Change web port
python app.py --port 9000
```
