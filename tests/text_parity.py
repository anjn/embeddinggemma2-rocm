import argparse, json
from pathlib import Path
import numpy as np
from embeddinggemma2_rocm import Embedder

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
p.add_argument("--output")
p.add_argument("--reference")
a = p.parse_args()
root = Path(a.bundle)
ref = np.load(
    Path(a.reference) if a.reference else root / "exports/text-native/reference.npz"
)
with Embedder(root) as model:
    out = model.encode_tokens(ref["ids"], ref["mask"])
    np.testing.assert_allclose(out, ref["embeddings"], atol=2e-4, rtol=2e-4)
    if "text" in ref:
        via_text = model.encode_texts([str(ref["text"])], prompt="query")
        np.testing.assert_allclose(via_text, ref["embeddings"], atol=2e-4, rtol=2e-4)
    texts = ["hello", "日本語", "naïve café", "👩‍💻"]
    # A batch can be split over a smaller prepared native profile.
    out = model.encode_texts(texts)
    reverse = model.encode_texts(texts[::-1])
    np.testing.assert_allclose(out, reverse[::-1], atol=1e-6, rtol=1e-5)
    assert np.isfinite(out).all() and np.allclose(
        np.linalg.norm(out, axis=1), 1, atol=1e-3
    )
    record = dict(
        passed=True,
        reference_max_abs=float(
            np.abs(
                model.encode_tokens(ref["ids"], ref["mask"]) - ref["embeddings"]
            ).max()
        ),
        last_execution=model.capabilities["last_execution"],
    )
    (Path(a.output) if a.output else root / "text-parity.json").write_text(
        json.dumps(record, indent=2)
    )
    print(record)
