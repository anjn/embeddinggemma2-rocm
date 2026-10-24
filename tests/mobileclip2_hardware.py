"""Hardware regression/benchmark against the prior NVR evaluation artifacts.

Run after test_mobileclip2.py with the same native runtime environment.
Inputs are local reviewed fixtures; nothing is downloaded or sent externally.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from embeddinggemma2_rocm import MobileCLIP2Embedder
from PIL import Image


def cosine(a, b):
    return np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )


def benchmark(fn):
    start = time.perf_counter()
    n = 0
    while n < 30 or time.perf_counter() - start < 2:
        fn()
        n += 1
    times = []
    for _ in range(100):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "iterations": 100,
        "warmups": n,
    }


def quality(images, texts, dataset):
    labels = np.array(list(dataset["labels"].values()))
    targets = np.array([q[2] for q in dataset["queries"]])
    n = len(targets)
    result = {}
    images = images / np.linalg.norm(images, axis=1, keepdims=True)
    texts = texts / np.linalg.norm(texts, axis=1, keepdims=True)
    for lang, t in [("en", texts[:n]), ("ja", texts[n:])]:
        order = np.argsort(-(t @ images.T), axis=1)
        rel = labels[order] == targets[:, None]
        ap = (np.cumsum(rel, axis=1) / np.arange(1, len(labels) + 1) * rel).sum(1) / (
            labels[None, :] == targets[:, None]
        ).sum(1)
        result[lang] = {
            "hit1": float(rel[:, 0].mean()),
            "hit5": float(rel[:, :5].any(1).mean()),
            "mAP": float(ap.mean()),
        }
    sim = images @ images.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, :-1]
    rel = labels[order] == labels[:, None]
    result["image_loo_hit1"] = float(rel[:, 0].mean())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["gpu", "npu", "both"], default="both")
    a = p.parse_args()
    r = a.reference
    dataset = json.loads((r / "dataset.json").read_text())
    crops = [
        np.asarray(Image.open(a.crops / f).convert("RGB")) for f in dataset["labels"]
    ]
    x = np.load(r / "inputs-letterbox.npy")
    fp32 = np.load(r / "images-letterbox.npy")
    queries = [q[0] for q in dataset["queries"]] + [q[1] for q in dataset["queries"]]
    results = {}
    for backend in ["npu", "gpu"] if a.backend == "both" else [a.backend]:
        with MobileCLIP2Embedder(a.bundle, backend) as model:
            model.prepare()
            text = model.encode_texts(queries)
            text_cos = cosine(text, np.load(r / "texts.npy"))
            assert text_cos.min() > 0.9999
            for batch in [1, 4] if backend == "gpu" else [1]:
                name = f"{backend}-b{batch}"
                out = []
                pre = []
                for i in range(0, len(x), batch):
                    out.append(model.encode_pixels(x[i : i + batch]))
                    pre.append(model.encode_images(crops[i : i + batch]))
                out = np.concatenate(out)
                pre = np.concatenate(pre)
                reference = np.load(
                    r
                    / (f"gpu-noalgebra-b{batch}" if backend == "gpu" else "npu-b1")
                    / "all-images.npy"
                )
                c = cosine(out, reference)
                assert c.min() > 0.999, (name, "backend drift", c.min())
                pc = cosine(out, pre)
                assert pc.min() > 0.99999, (name, "preprocessing drift", pc.min())
                timing = benchmark(lambda batch=batch: model.encode_pixels(x[:batch]))
                timing["per_image_ms"] = timing["median_ms"] / batch
                timing["fps"] = batch * 1000 / timing["median_ms"]
                full = benchmark(lambda batch=batch: model.encode_images(crops[:batch]))
                v = {
                    "inference": timing,
                    "with_preprocessing": full,
                    "prior_backend_cosine_min": float(c.min()),
                    "preprocessing_cosine_min": float(pc.min()),
                    "fp32_cosine_mean": float(cosine(out, fp32).mean()),
                    "text_fp32_cosine_min": float(text_cos.min()),
                    "quality": quality(out, text, dataset),
                }
                results[name] = v
                print(name, json.dumps(v), flush=True)
            results[backend + "-text"] = benchmark(
                lambda: model.encode_texts(queries[:1])
            )
    a.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
