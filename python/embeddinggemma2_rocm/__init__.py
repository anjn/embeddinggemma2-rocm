"""AMD GPU image/text embeddings through the native C ABI."""

from .runtime import Embedder, UnsupportedProfile

__all__ = ["Embedder", "UnsupportedProfile"]
__version__ = "0.1.0"
