import argparse, json
from pathlib import Path
import numpy as np
from embeddinggemma2_rocm import Embedder
from embeddinggemma2_rocm.preparation import normalized_aspect, fixture_size

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
p.add_argument("--budget", type=int, required=True)
p.add_argument("--aspect", nargs=2, type=int, default=[16, 9])
a = p.parse_args()
a.aspect = normalized_aspect(a.aspect)
root = Path(a.bundle)
reference = np.load(
    root / f"exports/budget{a.budget}-aspect{a.aspect[0]}x{a.aspect[1]}/reference.npy"
)
width, height = fixture_size(a.aspect)
rgb = np.empty((1, height, width, 3), np.uint8)
rgb[:] = [127, 95, 63]
with Embedder(root) as model:
    out = model.encode_images(rgb, token_budget=a.budget)
    cos = float((out * reference).sum())
    assert cos > 0.995 and np.isfinite(out).all()
    print(
        dict(
            budget=a.budget,
            min_cosine_to_fp32=cos,
            max_abs=float(np.abs(out - reference).max()),
            capabilities=model.capabilities,
        )
    )
