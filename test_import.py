try:
    from camera_utils import create_camera_capture, release_camera
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
