"""Explicit GPU lifecycle, routing, and validation checks on a prepared bundle."""

import argparse
import numpy as np
from embeddinggemma2_rocm import Embedder, UnsupportedProfile

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
a = p.parse_args()
for _ in range(3):
    with Embedder(a.bundle) as model:
        rgb = np.zeros((3, 288, 512, 3), np.uint8)
        first = model.encode_images(rgb)
        text = model.encode_texts(["hello", "日本語", "a cat"])
        np.testing.assert_allclose(
            model.encode_images(rgb), first, atol=1e-6, rtol=1e-5
        )
        np.testing.assert_allclose(
            model.encode_texts(["hello", "日本語", "a cat"]), text, atol=1e-6, rtol=1e-5
        )
        try:
            model.encode_tokens([[-1]], [[1]])
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid token id accepted")
        try:
            model.encode_tokens([[1]], [[0]])
        except ValueError:
            pass
        else:
            raise AssertionError("Empty attention mask accepted")
    model.close()
    try:
        model.encode_images(rgb)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Closed context accepted inference")
print(
    "Three native context lifecycles, mixed-modality routing and invalid token checks passed"
)
