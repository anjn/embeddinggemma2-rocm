"""GPU regression: shared image/text parity and stable allocations across switches."""

import argparse
import json
import numpy as np
from embeddinggemma2_rocm import Embedder

p = argparse.ArgumentParser()
p.add_argument("--original", required=True)
p.add_argument("--shared", required=True)
p.add_argument("--iterations", type=int, default=100)
a = p.parse_args()
rgb = np.random.default_rng(927).integers(0, 256, (4, 288, 512, 3), dtype=np.uint8)
cases = [("image", 1), ("image", 4), ("text", 1), ("text", 4)]


def encode(model, case):
    kind, batch = case
    return (
        model.encode_images(rgb[:batch])
        if kind == "image"
        else model.encode_texts(["日本語の検索"] * batch)
    )


with Embedder(a.shared) as model:
    # Initialize the shared context first: no Python torch/CUDA warmup is allowed.
    model.prepare()
    expected = {case: encode(model, case) for case in cases}
    # Reserve RGB capacity and eager text scratch for the actual workload first.
    for case in cases:
        np.testing.assert_allclose(
            encode(model, case), expected[case], atol=1e-6, rtol=1e-5
        )
    initial = model.capabilities["memory"]
    assert initial["mode"] == "shared"
    for i in range(a.iterations):
        case = cases[i % len(cases)]
        np.testing.assert_allclose(
            encode(model, case), expected[case], atol=1e-6, rtol=1e-5
        )
    final = model.capabilities["memory"]
    for field in (
        "weight_bytes",
        "weight_allocations",
        "workspace_bytes",
        "rgb_capacity_bytes",
        "transient_allocated_bytes",
        "transient_reserved_bytes",
    ):
        assert initial[field] == final[field], (field, initial[field], final[field])

with Embedder(a.original) as model:
    for case in cases:
        np.testing.assert_allclose(
            encode(model, case), expected[case], atol=1e-6, rtol=1e-5
        )

print(
    json.dumps(
        dict(
            passed=True,
            cold_shared_start=True,
            iterations=a.iterations,
            initial=initial,
            final=final,
        ),
        indent=2,
    )
)
