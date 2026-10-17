"""Check mixed-batch sequence separation within BF16/FP16 numeric tolerances."""

import argparse
import json
from pathlib import Path
import numpy as np
from embeddinggemma2_rocm.llama import LlamaBackend

p = argparse.ArgumentParser()
p.add_argument("--library", required=True)
p.add_argument("--model", required=True)
p.add_argument("--mmproj", required=True)
p.add_argument("--budget", type=int, default=70)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--flash", choices=["on", "off"], default="on")
p.add_argument("--images", help="Optional uint8 RGB NPY containing distinct images")
p.add_argument("--reference", help="Optional original FP32 retrieval reference NPZ")
p.add_argument("--output", required=True)
a = p.parse_args()
rgb = (
    np.load(a.images)[: a.batch]
    if a.images
    else np.random.default_rng(812).integers(
        0, 256, (a.batch, 288, 512, 3), dtype=np.uint8
    )
)
assert len(rgb) == a.batch
model = LlamaBackend(a.library, a.model, a.mmproj, a.budget, a.batch, a.flash == "on")
try:
    out = model.encode(rgb)
    reverse = model.encode(rgb[::-1])
    metadata = dict(model.metadata)
finally:
    model.close()
model = LlamaBackend(a.library, a.model, a.mmproj, a.budget, 1, a.flash == "on")
try:
    single = np.concatenate([model.encode(row[None]) for row in rgb[: min(a.batch, 4)]])
finally:
    model.close()
cos = np.sum(single * out[: len(single)], axis=1)
reverse_cos = np.sum(out * reverse[::-1], axis=1)
report = dict(
    batch=a.batch,
    budget=a.budget,
    min_cosine_to_single=float(cos.min()),
    reversal_min_cosine=float(reverse_cos.min()),
    reversal_max_abs=float(np.abs(out - reverse[::-1]).max()),
    metadata=metadata,
)
report["cosine_threshold"] = 0.999
report["absolute_threshold"] = 0.005
report["passed"] = bool(
    cos.min() >= 0.999
    and reverse_cos.min() >= 0.999
    and report["reversal_max_abs"] <= 0.005
)
if a.reference:
    reference = np.load(a.reference)["images"][: a.batch]
    report["min_cosine_to_original_fp32"] = float(np.sum(out * reference, axis=1).min())
Path(a.output).write_text(json.dumps(report, indent=2))
print(report)
assert report["passed"], report
