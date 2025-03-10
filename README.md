# Real-time People Counter with YOLOv8

A computer vision system that uses YOLOv8 to detect and count people in real-time from camera feeds. The system provides both a standalone application and a web interface for monitoring.

![People Counter Demo](https://via.placeholder.com/800x400?text=People+Counter+Demo)

## Features

- 🎯 Real-time people detection and counting using YOLOv8
- 📊 Live statistics (current count, maximum count, FPS)
- 📱 Web interface for remote monitoring
- 🖥️ GPU acceleration with automatic fallback to CPU
- 📷 Optimized camera stream handling
- 📈 Performance monitoring and statistics

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster inference)
- Camera with RTSP stream or webcam

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/camera-detection.git
   cd camera-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your camera connection in `camera_utils.py`

4. Download YOLOv8 model (if not included) and update path in `config.py`

## Usage

### Standalone Application

Run the detection script with display:

```bash
python detect.py
```

Run without display (headless mode):

```bash
python detect.py --no-display
```

Force CPU usage even if GPU is available:

```bash
python detect.py --cpu
```

### Web Interface

Start the web server:

```bash
python app.py
```

Access the web interface at: http://localhost:8888

## Docker Deployment

This application can be containerized for production deployment using Docker.

### Prerequisites

- Docker
- Docker Compose
- NVIDIA Container Toolkit (for GPU support)

### CPU-only Deployment

1. Configure your environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your camera URL and other settings
   ```

2. Build and start the container:

   ```bash
   docker-compose up -d
   ```

3. Access the web interface at: http://localhost:8888

### GPU-enabled Deployment

For GPU acceleration:

1. Configure your environment as above.

2. Ensure the NVIDIA Container Toolkit is installed:

   ```bash
   # On Ubuntu
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. Build and start the GPU container:

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
   ```

### Scaling and Production Considerations

- **Reverse Proxy**: For production, place behind Nginx or Traefik for SSL termination and security
- **Authentication**: Add authentication for the web interface
- **Logging**: Configure proper logging with volume mounts for persistence
- **Health Checks**: Implement health monitoring for the container

### Updating the Application

To update the application with new code:

```bash
git pull
docker-compose build
docker-compose up -d
```

## Architecture

The system is built with the following components:

1. **People Detector Class**: Manages detection in a threaded environment for optimal performance
2. **Flask Web Server**: Provides HTTP endpoints for the frontend
3. **Web Interface**: Shows real-time video and statistics
4. **Camera Utils**: Handles camera connection and stream processing
5. **GPU Utils**: Optimizes performance based on available hardware

## Configuration

The system can be configured through the following files:

- `config.py`: Model paths and detection thresholds
- `camera_utils.py`: Camera connection parameters
- `gpu_utils.py`: GPU optimization settings

## Performance Optimization

The system automatically:

- Detects available GPU resources
- Optimizes batch size and precision for the available hardware
- Provides real-time performance metrics
- Gracefully handles camera disconnects

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

- This project uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- Built with [Flask](https://flask.palletsprojects.com/) web framework
- Uses [OpenCV](https://opencv.org/) for image processing
