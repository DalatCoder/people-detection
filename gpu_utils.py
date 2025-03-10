"""
Utility functions to detect and use GPU acceleration
"""
import logging
import platform
import subprocess
import sys

logger = logging.getLogger(__name__)

def check_gpu_support():
    """
    Check if the system has GPU support for deep learning frameworks
    
    Returns:
        dict: Information about detected GPUs and support status
    """
    gpu_info = {
        "has_gpu": False,
        "platform": platform.system(),
        "cuda_available": False,
        "mps_available": False,  # Apple Metal Performance Shaders
        "opencl_available": False,
        "device_name": None,
        "gpu_count": 0,
        "cuda_version": None,
        "recommended_backend": "cpu"
    }
    
    # Check for CUDA support via PyTorch (most common for deep learning)
    try:
        import torch
        gpu_info["torch_version"] = torch.__version__
        
        # Check for CUDA
        if torch.cuda.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["cuda_available"] = True
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["recommended_backend"] = "cuda"
            
            # Get CUDA version
            if hasattr(torch.version, 'cuda'):
                gpu_info["cuda_version"] = torch.version.cuda
                
            logger.info(f"CUDA is available with {gpu_info['gpu_count']} device(s)")
            logger.info(f"Device: {gpu_info['device_name']}")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["mps_available"] = True
            gpu_info["device_name"] = "Apple Silicon"
            gpu_info["recommended_backend"] = "mps"
            logger.info("Apple MPS is available for GPU acceleration")
    except ImportError:
        logger.warning("PyTorch not found, skipping CUDA/MPS detection")
    except Exception as e:
        logger.warning(f"Error checking PyTorch GPU support: {e}")
    
    # Check for OpenCL support via OpenCV
    try:
        import cv2
        # Check if OpenCV was built with OpenCL support
        if cv2.ocl.haveOpenCL():
            gpu_info["opencl_available"] = True
            if not gpu_info["has_gpu"]:  # Only set if no CUDA/MPS found
                gpu_info["has_gpu"] = True
                gpu_info["recommended_backend"] = "opencl"
            logger.info("OpenCL is available via OpenCV")
            
            # Enable OpenCL
            cv2.ocl.setUseOpenCL(True)
            logger.info(f"OpenCV OpenCL enabled: {cv2.ocl.useOpenCL()}")
    except Exception as e:
        logger.warning(f"Error checking OpenCV OpenCL support: {e}")
    
    # Check for NVIDIA GPU using system commands (fallback method)
    if not gpu_info["cuda_available"] and platform.system() in ['Linux', 'Windows']:
        try:
            if platform.system() == 'Windows':
                # Use Windows Management Instrumentation
                proc = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                    capture_output=True, text=True)
                if 'NVIDIA' in proc.stdout:
                    logger.info("NVIDIA GPU detected but CUDA not available in PyTorch")
                    gpu_info["has_gpu"] = True
                    gpu_info["device_name"] = "NVIDIA GPU (CUDA not enabled)"
            else:
                # Use nvidia-smi on Linux
                proc = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True)
                if proc.returncode == 0 and proc.stdout.strip():
                    logger.info(f"NVIDIA GPU detected: {proc.stdout.strip()}")
                    gpu_info["has_gpu"] = True
                    gpu_info["device_name"] = proc.stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    
    return gpu_info

def optimize_performance(gpu_info=None):
    """
    Optimize performance settings based on detected hardware
    """
    if gpu_info is None:
        gpu_info = check_gpu_support()
        
    # Set optimal thread settings
    try:
        import cv2
        if gpu_info["has_gpu"]:
            # Reduce CPU thread usage when GPU is available
            cv2.setNumThreads(4)
        else:
            # Use more CPU threads when no GPU
            cv2.setNumThreads(8)
        logger.info(f"OpenCV threads set to {cv2.getNumThreads()}")
        
        # Enable OpenCL if available and no CUDA
        if gpu_info["opencl_available"] and not gpu_info["cuda_available"]:
            cv2.ocl.setUseOpenCL(True)
            logger.info(f"OpenCV OpenCL enabled: {cv2.ocl.useOpenCL()}")
    except Exception as e:
        logger.warning(f"Error optimizing OpenCV settings: {e}")
    
    # Configure PyTorch settings
    try:
        import torch
        if gpu_info["cuda_available"]:
            if not torch.backends.cudnn.enabled:
                torch.backends.cudnn.enabled = True
                logger.info("Enabled cuDNN for performance boost")
            # Set benchmark mode for optimized performance with fixed input sizes
            torch.backends.cudnn.benchmark = True
        elif gpu_info["mps_available"]:
            torch.backends.mps.enable_mps = True
            logger.info("Enabled MPS for Apple Silicon performance")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error optimizing PyTorch settings: {e}")
    
    return gpu_info

def select_device(gpu_info=None, force_cpu=False):
    """
    Select the optimal device for model inference
    
    Args:
        gpu_info: GPU info dict from check_gpu_support()
        force_cpu: Force CPU usage even if GPU is available
        
    Returns:
        str: Device string for PyTorch ('cuda:0', 'mps', or 'cpu')
    """
    if gpu_info is None:
        gpu_info = check_gpu_support()
        
    if force_cpu:
        logger.info("Forcing CPU usage as requested")
        return "cpu"
        
    if gpu_info["cuda_available"]:
        return "cuda:0"
    elif gpu_info["mps_available"]:
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Print detailed GPU information
    gpu_info = check_gpu_support()
    print("\n==== GPU Detection Results ====")
    print(f"Platform: {gpu_info['platform']}")
    print(f"GPU Available: {gpu_info['has_gpu']}")
    
    if gpu_info['has_gpu']:
        print(f"Device Name: {gpu_info['device_name']}")
        
        if gpu_info['cuda_available']:
            print(f"CUDA Available: Yes (version {gpu_info['cuda_version']})")
            print(f"GPU Count: {gpu_info['gpu_count']}")
        elif gpu_info['mps_available']:
            print("Apple MPS Available: Yes")
        elif gpu_info['opencl_available']:
            print("OpenCL Available: Yes")
            
    print(f"Recommended Backend: {gpu_info['recommended_backend'].upper()}")
    print("==============================\n")
    
    # Optimize based on results
    optimize_performance(gpu_info)
    
    # Get the recommended device
    device = select_device(gpu_info)
    print(f"Selected device: {device}")
