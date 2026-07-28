"""
Device management utility.
Selects the best available compute device (CUDA, MPS, or CPU) with
graceful fallbacks when a requested accelerator is unavailable.
"""

import logging
import torch

logger = logging.getLogger("thesis")


def _cuda_available() -> bool:
    """Robust CUDA availability check that tolerates missing drivers.

    torch.cuda.is_available() can be True in some environments even when the
    NVIDIA driver is not properly installed, which would crash at model.to().
    We attempt a lightweight CUDA call and fall back to CPU on failure.
    """
    try:
        if not torch.cuda.is_available():
            return False
        # Trigger lazy init to surface driver issues early
        _ = torch.cuda.current_device()
        _ = torch.cuda.get_device_name(0)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("CUDA reported available but initialization failed: %s. Using CPU.", exc)
        return False


def _mps_available() -> bool:
    """Robust MPS availability check for Apple Silicon."""
    try:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("MPS check failed (%s). Using CPU.", exc)
        return False


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available compute device.
    
    Args:
        preference: One of 'auto', 'cpu', 'cuda', 'mps'.
        
    Returns:
        torch.device object.
    """
    if preference == "auto":
        if _cuda_available():
            device = torch.device("cuda")
        elif _mps_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if preference == "cuda" and not _cuda_available():
            logger.warning("Requested CUDA but no GPU detected; falling back to CPU.")
            device = torch.device("cpu")
        elif preference == "mps" and not _mps_available():
            logger.warning("Requested MPS but it is unavailable; falling back to CPU.")
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
