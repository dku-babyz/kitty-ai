"""Unified inference package.

Usage
-----
>>> from inference.text import TextPredictor
>>> from inference.image import ImagePredictor
"""

from .base import get_device, seed_all
from .text import TextPredictor
from .image import ImagePredictor

__all__ = [
    "get_device", "seed_all",
    "TextPredictor", "ImagePredictor"
]