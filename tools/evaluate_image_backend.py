"""Compare a direct-call baseline's image embeddings with fixed FP32 text queries."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from evaluate_quality import recall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["torch", "llama"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mmproj")
    p.add_argument("--llama-library")
    p.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    p.add_argument("--precision", choices=["fp32", "mixed", "bf16"], default="mixed")
    p.add_argument("--flash", choices=["on", "off"], default="on")
    p.add_argument("--budget", type=int, choices=[70, 140, 280], default=70)
    p.add_argument("--batch", type=int, choices=[1, 4, 32, 128], default=4)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = json.loads(Path(a.manifest).read_text())["images"]
    if (
        len(rows) != 256
        or len({r["id"] for r in rows}) != 256
        or any(len(r["captions"]) != 5 for r in rows)
    ):
        raise ValueError("Expected all 256 distinct images and all 1280 captions")
    captions = [text for row in rows for text in row["captions"]]
    images = []
    digest = hashlib.sha256()
    for row in rows:
        with Image.open(Path(a.data_root) / row["image"]) as image:
            rgb = np.asarray(
                image.convert("RGB").resize((512, 288), Image.Resampling.BILINEAR)
            )
        images.append(rgb)
        digest.update(rgb.tobytes())
    digest.update(json.dumps(captions, ensure_ascii=False).encode())
    digest.update(str(a.budget).encode())
    reference = np.load(a.reference)
    if digest.hexdigest() != str(reference["input_sha256"]):
        raise ValueError("Reference inputs and budget do not match")
    images = np.stack(images)
    if a.backend == "torch":
        from embeddinggemma2_rocm.benchmark import TorchBackend

        model = TorchBackend(a.model, a.budget, a.attention, a.precision)
    else:
        from embeddinggemma2_rocm.llama import LlamaBackend

        model = LlamaBackend(
            a.llama_library, a.model, a.mmproj, a.budget, a.batch, a.flash == "on"
        )
    try:
        output = np.concatenate(
            [
                model.encode(images[start : start + a.batch])
                for start in range(0, 256, a.batch)
            ]
        )
        metadata = model.metadata
    finally:
        if hasattr(model, "close"):
            model.close()
    if (
        output.shape != (256, 768)
        or not np.isfinite(output).all()
        or not np.allclose(np.linalg.norm(output, axis=1), 1, atol=0.001)
    ):
        raise ValueError("Invalid normalized embeddings")
    labels = np.repeat(np.arange(256), 5)
    baseline = recall(reference["images"], reference["texts"], labels)
    candidate = recall(output, reference["texts"], labels)
    loss = {key: baseline[key] - candidate[key] for key in baseline}
    report = dict(
        mode="Image embeddings with fixed original FP32 text queries",
        backend=a.backend,
        budget=a.budget,
        batch=a.batch,
        images=256,
        captions=1280,
        input_sha256=digest.hexdigest(),
        reference=baseline,
        candidate=candidate,
        loss_percentage_points=loss,
        min_cosine_to_fp32=float(np.sum(output * reference["images"], axis=1).min()),
        passed=all(value <= 1.0 for value in loss.values()),
        metadata=metadata,
    )
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    np.save(Path(a.output).with_suffix(".npy"), output)


if __name__ == "__main__":
    main()
