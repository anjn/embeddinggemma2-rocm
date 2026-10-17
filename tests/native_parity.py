"""Explicit GPU parity/performance check against a supplied reference output."""

import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
from embeddinggemma2_rocm import Embedder

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
p.add_argument("--images", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--output", required=True)
p.add_argument("--repeat", type=int, default=5)
p.add_argument("--atol", type=float, default=1e-6)
p.add_argument("--rtol", type=float, default=1e-5)
p.add_argument("--min-cosine", type=float, default=0.995)
a = p.parse_args()
x = np.load(a.images)
ref = np.load(a.reference)
with Embedder(a.bundle) as model:
    for _ in range(3):
        out = model.encode_images(x)
    times = []
    for _ in range(a.repeat):
        t = time.perf_counter()
        out = model.encode_images(x)
        times.append(time.perf_counter() - t)
    rev = model.encode_images(x[::-1].copy())
np.testing.assert_allclose(out, ref, atol=a.atol, rtol=a.rtol)
assert float(np.sum(out * ref, axis=1).min()) >= a.min_cosine
np.testing.assert_array_equal(rev, out[::-1])
assert np.isfinite(out).all() and np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-3)
r = dict(
    passed=True,
    atol=a.atol,
    rtol=a.rtol,
    cosine_threshold=a.min_cosine,
    images=len(x),
    input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
    max_abs=float(np.abs(out - ref).max()),
    min_cosine=float(np.sum(out * ref, axis=1).min()),
    seconds=times,
    ms_per_image=float(np.median(times) * 1000 / len(x)),
    reversal_max_abs=float(np.abs(rev - out[::-1]).max()),
)
Path(a.output).write_text(json.dumps(r, indent=2))
print(r)
