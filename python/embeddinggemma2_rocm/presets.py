"""Portable tuning recipes. Never execute a search during normal preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .preparation import MODEL_REVISION

SCHEMA = 1


def recipe_digest():
    root = Path(__file__).parent / "_compiler"
    h = hashlib.sha256()
    for p in sorted([*root.glob("*.py"), *root.glob("kernels/*.cpp")]):
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def environment():
    import migraphx
    import torch

    from .runtime import device_info

    device = device_info()
    return dict(
        device={k: device[k] for k in ("arch", "name", "wave_size", "compute_units")},
        limits={
            k: device[k]
            for k in (
                "max_threads_per_block",
                "shared_memory_per_block",
                "total_memory",
            )
        },
        software=dict(
            torch=torch.__version__,
            rocm=torch.version.hip,
            migraphx=getattr(migraphx, "__version__", "unknown"),
        ),
        recipe_sha256=recipe_digest(),
    )


def profile_key(budget, batch, aspect, precision):
    from .preparation import normalized_aspect

    return dict(
        model_revision=MODEL_REVISION,
        budget=budget,
        batch=batch,
        aspect=list(normalized_aspect(aspect)),
        precision=precision,
    )


def registry():
    source = Path(__file__).resolve().parents[2] / "profiles/gpus"
    return source if source.is_dir() else Path(__file__).parent / "profiles/gpus"


def validate(record):
    if record.get("schema") != SCHEMA or record.get("kind") != "hip_tuning_recipe":
        raise ValueError("Unsupported tuning recipe schema")
    from ._compiler.search import candidates

    config = record["kernel_config"]
    if config is not None and (
        not isinstance(config, dict)
        or set(config)
        - {
            "gelu",
            "qk",
            "norm",
            "preprocess",
            "segmented",
        }
    ):
        raise ValueError("Unknown kernel family")
    if config is not None and type(config.get("segmented")) is not bool:
        raise ValueError("segmented must be a boolean")
    batch = record["profile"]["batch"]
    vb = record["vision_batch"]
    if (
        type(vb) is not int
        or vb < 1
        or type(batch) is not int
        or batch < 1
        or batch % vb
    ):
        raise ValueError("Vision batch must divide the execution batch")
    env = record["environment"]
    device = dict(env["device"], **env["limits"])
    for family in ("gelu", "qk", "norm", "preprocess"):
        c = (config or {}).get(family)
        if c is not None and c not in candidates(family, device):
            raise ValueError(f"Illegal {family} configuration for the measured device")
    validation = record.get("validation", {})
    if not validation.get("numeric_pass") or not validation.get("retrieval_pass"):
        raise ValueError("Recipe has not passed numerical and retrieval validation")
    import math

    cosine = validation.get("min_cosine_to_standard", float("nan"))
    if not math.isfinite(cosine) or cosine < 0.995:
        raise ValueError("Numerical validation is insufficient")
    metrics = {"text_to_image", "image_to_text"}
    losses = validation.get("loss_percentage_points", {})
    retrieval = validation.get("retrieval", {})
    if set(losses) != metrics or set(retrieval) != metrics:
        raise ValueError("Both retrieval directions must be validated")
    if not all(math.isfinite(v) and -100 <= v <= 1 for v in losses.values()):
        raise ValueError("Invalid retrieval loss")
    if not all(math.isfinite(v) and 0 <= v <= 100 for v in retrieval.values()):
        raise ValueError("Invalid retrieval recall")
    return record


def select(key, env, explicit=None):
    paths = [Path(explicit)] if explicit else sorted(registry().glob("**/*.json"))
    reasons = []
    for path in paths:
        try:
            r = validate(json.loads(path.read_text()))
            if r["profile"] != key:
                raise ValueError("profile does not match")
            if r["environment"]["device"] != env["device"]:
                raise ValueError("GPU does not match")
            if r["environment"]["software"] != env["software"]:
                raise ValueError("software versions are unvalidated")
            if r["environment"]["recipe_sha256"] != env["recipe_sha256"]:
                raise ValueError("recipe source changed")
            return r
        except (ValueError, KeyError, TypeError) as e:
            if explicit:
                raise ValueError(f"Cannot apply tuning recipe {path}: {e}") from e
            reasons.append(str(e))
    print(
        "No compatible published tuning recipe; using the compiled default. "
        + (
            "; ".join(sorted(set(reasons)))
            if reasons
            else "No measured recipe is installed."
        ),
        flush=True,
    )
    return None


def export_recipe(profile, quality, output):
    from ._bundle import bundle_digest
    from .preparation import write_json

    report = json.loads(Path(profile).read_text())
    q = json.loads(Path(quality).read_text())
    best = report.get("selected")
    if not best or not report.get("complete"):
        raise ValueError("A completed search with a selected candidate is required")
    if not q.get("passed") or (q.get("images"), q.get("captions")) != (256, 1280):
        raise ValueError("Complete passing retrieval validation is required")
    if (
        Path(q.get("bundle", "")).resolve() != Path(best["bundle"]).resolve()
        or q.get("budget") != report["budget"]
    ):
        raise ValueError("Quality report does not identify the selected candidate")
    if q.get("bundle_sha256") != bundle_digest(best["bundle"]):
        raise ValueError("Candidate changed after quality validation")
    root = Path(best["bundle"])
    index = json.loads((root / "index.json").read_text())
    key = report["profile"]
    matches = [
        p
        for p in index["profiles"]
        if p["budget"] == key["budget"]
        and p["batch"] == key["batch"]
        and p["aspect"] == key["aspect"]
        and p["precision"] == key["precision"]
    ]
    if len(matches) != 1 or index["revision"] != key["model_revision"]:
        raise ValueError("Search profile does not identify the validated image profile")
    bundle = json.loads((root / matches[0]["path"] / "bundle.json").read_text())
    if (
        bundle["vision"]["batch"] != best["vision_batch"]
        or bundle["fingerprint"].get("kernel_config") != best["kernel_config"]
    ):
        raise ValueError("Search settings differ from the validated bundle")
    if set(q["loss_percentage_points"]) != {
        "text_to_image",
        "image_to_text",
    } or not all(float(v) <= 1 for v in q["loss_percentage_points"].values()):
        raise ValueError("Retrieval loss exceeds one percentage point")
    out = Path(output)
    if out.exists():
        raise ValueError("Use a new recipe file")
    record = dict(
        schema=SCHEMA,
        kind="hip_tuning_recipe",
        environment=report["environment"],
        profile=report["profile"],
        vision_batch=best["vision_batch"],
        kernel_config=best["kernel_config"],
        measurement=dict(
            ms_per_image=best["ms_per_image"],
            input_sha256=report.get("input_sha256"),
            search_sha256=hashlib.sha256(Path(profile).read_bytes()).hexdigest(),
            search=report.get("search"),
        ),
        validation=dict(
            numeric_pass=True,
            numeric_reference=report.get("reference_backend", "compiled_standard"),
            retrieval_pass=True,
            min_cosine_to_standard=best["min_cosine_to_standard"],
            retrieval=q["candidate"],
            reference_retrieval=q["reference"],
            images=q["images"],
            captions=q["captions"],
            manifest_sha256=q["manifest_sha256"],
            loss_percentage_points=q["loss_percentage_points"],
            input_sha256=q["input_sha256"],
        ),
    )
    validate(record)
    write_json(out, record)
    return out
