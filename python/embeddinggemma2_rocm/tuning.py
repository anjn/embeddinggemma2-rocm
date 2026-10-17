"""Per-GPU whole-pipeline search. Candidate activation is an explicit operation."""

from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from .preparation import compile_images, export_images, write_json


def _measure(bundle, budget, batch, path, repeat):
    cmd = [
        sys.executable,
        "-m",
        "embeddinggemma2_rocm",
        "benchmark",
        "--backend",
        "native",
        "--bundle",
        str(bundle),
        "--budget",
        str(budget),
        "--batch",
        str(batch),
        "--warmup",
        "3",
        "--repeat",
        str(repeat),
        "--output",
        str(path),
    ]
    with path.with_suffix(".log").open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if not path.exists():
        write_json(path, dict(status="process_error", returncode=result.returncode))
    return json.loads(path.read_text())


def tune(a):
    if a.budget == 280 and a.batch > 32:
        raise ValueError("Budget 280 tuning supports request batches up to 32")
    output = Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    exports = output / "exports"
    export_images(a.model, exports, [a.budget], (16, 9))
    session = output / "tuning" / str(time.time_ns())
    session.mkdir(parents=True)
    roots = []
    for vb in sorted(set(a.vision_batches)):
        execution = min(a.batch, 32) if a.budget == 280 else a.batch
        if vb < 1 or vb > execution or execution % vb:
            continue
        dest = session / f"vision{vb}"
        dest.mkdir()
        (dest / "exports").symlink_to(exports, target_is_directory=True)
        try:
            compile_images(
                a.model,
                dest,
                [a.budget],
                [a.batch],
                (16, 9),
                "fast",
                vb,
                cache_dirs=[output, *[candidate for _, candidate in roots]],
            )
            roots.append((vb, dest))
        except Exception as error:
            write_json(
                dest / "preparation-error.json",
                dict(status="compile_error", error=str(error)),
            )
    # A separately compiled standard-precision reference is measured once for numeric checks.
    reference = session / "reference"
    reference.mkdir()
    (reference / "exports").symlink_to(exports, target_is_directory=True)
    compile_images(
        a.model,
        reference,
        [a.budget],
        [a.batch],
        (16, 9),
        "standard",
        min(a.batch, 4),
        cache_dirs=[output, *[candidate for _, candidate in roots]],
    )
    ref_result = _measure(reference, a.budget, a.batch, session / "reference.json", 1)
    if ref_result.get("status") != "ok":
        raise RuntimeError("Standard GPU reference failed; no profile can be selected")
    ref = np.load(session / "reference.npy")
    records = []
    for round_, order in enumerate((roots, roots[::-1])):
        for vb, dest in order:
            path = session / f"v{vb}-round{round_}.json"
            result = _measure(dest, a.budget, a.batch, path, a.repeat)
            result.update(vision_batch=vb, round=round_)
            if result.get("status") == "ok":
                x = np.load(path.with_suffix(".npy"))
                cos = (x * ref).sum(1) / (
                    np.linalg.norm(x, axis=1) * np.linalg.norm(ref, axis=1)
                )
                result["min_cosine_to_standard"] = float(cos.min())
                result["numeric_pass"] = bool(cos.min() >= 0.995)
            records.append(result)
            write_json(session / "measurements.json", records)
    candidates = []
    for vb, dest in roots:
        group = [r for r in records if r["vision_batch"] == vb]
        if len(group) == 2 and all(
            r.get("status") == "ok" and r.get("numeric_pass") for r in group
        ):
            times = [t for r in group for t in r["seconds"]]
            candidates.append(
                dict(
                    vision_batch=vb,
                    bundle=str(dest),
                    ms_per_image=float(np.median(times) * 1000 / a.batch),
                    min_cosine_to_standard=min(
                        r["min_cosine_to_standard"] for r in group
                    ),
                )
            )
    best = min(candidates, key=lambda r: r["ms_per_image"]) if candidates else None
    report = dict(
        schema=1,
        budget=a.budget,
        batch=a.batch,
        selected=best,
        candidates=candidates,
        activated=False,
        quality_status="numeric calibration only; retrieval validation required before activation",
        measurements=records,
    )
    write_json(session / "profile.json", report)
    print(json.dumps(report, indent=2))
    if best is None:
        raise RuntimeError(
            "No candidate passed both reversed rounds and numeric calibration"
        )
    return report
