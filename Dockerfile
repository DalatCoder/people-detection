FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for models if it doesn't exist
RUN mkdir -p models

# Download YOLOv8 model if not present
RUN python -c "import os; from ultralytics import YOLO; \
    os.environ['PYTHONPATH'] = '.'; \
    if not os.path.exists('models/yolov8n.pt'): \
        YOLO('yolov8n.pt').save('models/yolov8n.pt')"

# Update config to use the downloaded model
RUN echo "MODEL_CONFIG = {'model_path': 'models/yolov8n.pt', 'conf_threshold': 0.25}" > config.py

# Expose the port the app runs on
EXPOSE 8888

# Command to run the application
CMD ["python", "app.py"]
```
