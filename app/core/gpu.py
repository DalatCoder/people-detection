"""
GPU utilities for optimizing performance.
"""
import os
import torch

def check_gpu_support():
    """
    Check for available GPU support and return information.
    
    Returns:
        dict: GPU information including availability
    """
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_names': []
    }
    
    if gpu_info['cuda_available']:
        for i in range(gpu_info['device_count']):
            gpu_info['device_names'].append(torch.cuda.get_device_name(i))
    
    return gpu_info

def select_device(gpu_info, force_cpu=False):
    """
    Select the appropriate device (CPU or GPU) based on availability and preferences.
    
    Args:
        gpu_info: GPU information from check_gpu_support
        force_cpu: Force CPU usage even if GPU is available
        
    Returns:
        str: Device name to use
    """
    if force_cpu:
        return 'cpu'
    
    if gpu_info['cuda_available'] and gpu_info['device_count'] > 0:
        return 'cuda:0'  # Use first GPU
    
    return 'cpu'

def optimize_performance(gpu_info):
    """
    Apply performance optimizations based on available hardware.
    
    Args:
        gpu_info: GPU information from check_gpu_support
    """
    # Set environment variables for better performance
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Apply GPU-specific optimizations
    if gpu_info['cuda_available']:
        # Allow TensorFloat32 format for faster computation on Ampere GPUs
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable memory efficient operations
        torch.cuda.empty_cache()
        
        print(f"GPU acceleration enabled. Found {gpu_info['device_count']} device(s):")
        for i, name in enumerate(gpu_info['device_names']):
            print(f"  [{i}] {name}")
    else:
        print("No GPU detected. Using CPU for computation.")
