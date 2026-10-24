"""AMD GPU image/text embeddings through the native C ABI."""

from .runtime import Embedder, UnsupportedProfile
from .mobileclip2 import MobileCLIP2Embedder

__all__ = ["Embedder", "UnsupportedProfile", "MobileCLIP2Embedder"]
__version__ = "0.1.0"
