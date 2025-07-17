"""Common utilities shared across inference modules."""
import random, os, numpy as np, torch

__all__ = ["get_device", "seed_all"]

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available device (cuda ▸ mps ▸ cpu)."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    # Apple's Metal Performance Shaders (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def seed_all(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Inference‑time global settings
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True