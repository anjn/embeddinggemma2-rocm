"""GPU parity and latency regression for a selected shared image pipeline."""

import argparse
import json
import time
import numpy as np
from embeddinggemma2_rocm import Embedder
from embeddinggemma2_rocm.benchmark import images
from embeddinggemma2_rocm.preparation import write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--legacy", required=True)
p.add_argument("--shared", required=True)
p.add_argument("--output", required=True)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--budget", type=int, choices=[70, 140, 280], default=70)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--repeat", type=int, default=5)
a = p.parse_args()
if not 1 <= a.batch <= 128 or a.warmup < 1 or a.repeat < 1:
    p.error("Invalid batch or measurement count")
rgb = images(a.batch, unique=True)
results, outputs = {}, {}
for label, bundle in [("legacy", a.legacy), ("shared", a.shared)]:
    print("Testing", label, flush=True)
    with Embedder(bundle) as model:
        for _ in range(a.warmup):
            model.encode_images(rgb, token_budget=a.budget)
        elapsed = []
        for _ in range(a.repeat):
            start = time.perf_counter()
            out = model.encode_images(rgb, token_budget=a.budget)
            elapsed.append(time.perf_counter() - start)
        outputs[label] = out
        capabilities = model.capabilities
        results[label] = dict(
            seconds=elapsed,
            ms_per_image=float(np.median(elapsed) * 1000 / a.batch),
            memory=capabilities["memory"],
            execution=capabilities["last_execution"],
        )
        print(label, results[label]["ms_per_image"], flush=True)
ratio = results["shared"]["ms_per_image"] / results["legacy"]["ms_per_image"]
report = dict(
    parity_passed=bool(
        np.allclose(outputs["shared"], outputs["legacy"], atol=1e-6, rtol=1e-5)
    ),
    budget=a.budget,
    batch=a.batch,
    unique_inputs=True,
    warmup=a.warmup,
    repeat=a.repeat,
    max_absolute_error=float(np.max(abs(outputs["shared"] - outputs["legacy"]))),
    min_cosine=float((outputs["shared"] * outputs["legacy"]).sum(1).min()),
    shared_over_legacy_latency=ratio,
    within_five_percent=ratio <= 1.05,
    results=results,
)
write_json(a.output, report)
print(json.dumps(report, indent=2), flush=True)
if not report["parity_passed"] or not report["within_five_percent"]:
    raise SystemExit("Shared pipeline parity or latency regression")
