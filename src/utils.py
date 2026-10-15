import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: The integer seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detects and returns the best available computing device.
    
    Returns:
        torch.device: CUDA GPU if available, else MPS (Apple Silicon), else CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using CUDA GPU: {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def save_checkpoint(state: dict, filepath: str) -> None:
    """Saves a model checkpoint dictionary to disk.
    
    Args:
        state: State dictionary containing model state_dict, optimizer, epoch, metrics.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"[Checkpoint] Successfully saved to {filepath}")
