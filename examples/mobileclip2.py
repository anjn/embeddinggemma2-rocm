"""Usage: python examples/mobileclip2.py BUNDLE gpu|npu IMAGE [IMAGE ...]."""

import sys

import numpy as np
from embeddinggemma2_rocm import MobileCLIP2Embedder
from PIL import Image

with MobileCLIP2Embedder(sys.argv[1], backend=sys.argv[2]) as model:
    images = [np.asarray(Image.open(p).convert("RGB")) for p in sys.argv[3:]]
    vectors = model.encode_images(images)
    query = model.encode_texts("a person wearing a hat")
    print("vectors:", vectors.shape, "similarities:", (vectors @ query.T).ravel())
