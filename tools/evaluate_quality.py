"""Evaluate every image/caption pair; save reference embeddings separately from Git.

Manifest: {"images":[{"id":"unique", "image":"relative/path.png", "captions":[...]}]}.
Paths resolve against --data-root. Supply --model to generate the original FP32
reference once, or reuse --reference. No dataset is downloaded or redistributed.
"""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
from embeddinggemma2_rocm import Embedder
from embeddinggemma2_rocm._bundle import bundle_digest


def recall(images, texts, labels):
    scores = texts @ images.T
    return dict(
        text_to_image=float((scores.argmax(1) == labels).mean() * 100),
        image_to_text=float(
            (labels[scores.argmax(0)] == np.arange(len(images))).mean() * 100
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-only", action="store_true")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--model")
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=70, choices=[70, 140, 280])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    a = p.parse_args()
    manifest = Path(a.manifest)
    rows = json.loads(manifest.read_text())["images"]
    if len(rows) != 256 or len({r["id"] for r in rows}) != 256:
        raise ValueError("Release quality gate requires 256 distinct images")
    texts = [c for r in rows for c in r["captions"]]
    labels = np.array([i for i, r in enumerate(rows) for _ in r["captions"]])
    if len(texts) != 1280:
        raise ValueError("Release quality gate requires 1280 captions")
    images = []
    digest = hashlib.sha256()
    for row in rows:
        with Image.open(Path(a.data_root) / row["image"]) as image:
            rgb = np.asarray(
                image.convert("RGB").resize(
                    (a.width, a.height), Image.Resampling.BILINEAR
                )
            )
        digest.update(rgb.tobytes())
        images.append(rgb)
    digest.update(json.dumps(texts, ensure_ascii=False).encode())
    digest.update(str(a.budget).encode())
    input_hash = digest.hexdigest()
    images = np.stack(images)
    reference = Path(a.reference)
    if reference.exists():
        cached = np.load(reference)
        if str(cached["input_sha256"]) != input_hash:
            raise ValueError("Reference inputs/budget do not match")
        ref_images = cached["images"]
        ref_texts = cached["texts"]
    else:
        if not a.model:
            p.error("--model is required to create the reference")
        import torch
        from embeddinggemma2_rocm.preparation import load_model

        st = load_model(a.model)
        # Original FP32 CPU model, identical decoded/prescaled input and explicit budget.
        with torch.inference_mode():
            ref_images = []
            for rgb in images:
                features = st[0].preprocess(
                    [{"image": Image.fromarray(rgb)}],
                    prompt="",
                    processing_kwargs={"image": {"max_soft_tokens": a.budget}},
                )
                ref_images.append(st(features)["sentence_embedding"].float().numpy())
            ref_images = np.concatenate(ref_images)
            ref_texts = []
            for start in range(0, len(texts), 4):
                features = st[0].preprocess(
                    texts[start : start + 4], prompt="task: search result | query: "
                )
                ref_texts.append(st(features)["sentence_embedding"].float().numpy())
            ref_texts = np.concatenate(ref_texts)
        reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference, images=ref_images, texts=ref_texts, input_sha256=input_hash)
    if a.reference_only:
        print(
            json.dumps(
                dict(
                    reference=str(reference),
                    input_sha256=input_hash,
                    images=len(ref_images),
                    captions=len(ref_texts),
                )
            )
        )
        return
    if ref_images.shape != (256, 768) or ref_texts.shape != (1280, 768):
        raise ValueError("Invalid reference shape")
    with Embedder(a.bundle) as model:
        native_images = np.concatenate(
            [
                model.encode_images(images[i : i + a.batch], token_budget=a.budget)
                for i in range(0, len(images), a.batch)
            ]
        )
        image_execution = model.capabilities["last_execution"]
        native_texts = np.concatenate(
            [
                model.encode_texts(texts[i : i + a.batch], prompt="query")
                for i in range(0, len(texts), a.batch)
            ]
        )
        capabilities = model.capabilities
    for x in (native_images, native_texts, ref_images, ref_texts):
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite embedding")
    baseline = recall(ref_images, ref_texts, labels)
    candidate = recall(native_images, native_texts, labels)
    loss = {k: baseline[k] - candidate[k] for k in baseline}
    cosine = dict(
        images=float(np.sum(native_images * ref_images, axis=1).min()),
        texts=float(np.sum(native_texts * ref_texts, axis=1).min()),
    )
    record = dict(
        schema=1,
        bundle_sha256=bundle_digest(a.bundle),
        bundle=str(Path(a.bundle).resolve()),
        input_sha256=input_hash,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        images=256,
        captions=1280,
        budget=a.budget,
        requested_batch=a.batch,
        image_execution=image_execution,
        reference=baseline,
        candidate=candidate,
        loss_percentage_points=loss,
        min_cosine_to_fp32=cosine,
        criterion="Both retrieval R@1 losses are at most 1.0 percentage point",
        passed=all(v <= 1.0 for v in loss.values()),
        capabilities=capabilities,
    )
    Path(a.output).write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
