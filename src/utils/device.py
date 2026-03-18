"""
Device management utility.
Selects the best available compute device (CUDA, MPS, or CPU).
"""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available compute device.
    
    Args:
        preference: One of 'auto', 'cpu', 'cuda', 'mps'.
        
    Returns:
        torch.device object.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    return device


def get_device_info(device: torch.device) -> dict:
    """Get information about the compute device.
    
    Returns:
        Dictionary with device information.
    """
    info = {"device": str(device), "type": device.type}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        info["memory_total_gb"] = torch.cuda.get_device_properties(device).total_mem / 1e9
        info["cuda_version"] = torch.version.cuda
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"

    info["pytorch_version"] = torch.__version__
    return info
