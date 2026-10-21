"""Resumable finite HIP screening followed by whole-pipeline validation."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .preparation import export_images, write_json


def _read_checkpoint(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        saved = path.with_name(path.name + f".corrupt-{time.time_ns()}")
        path.replace(saved)
        print(
            f"Incomplete checkpoint preserved at {saved}; repeating this step",
            flush=True,
        )
        return default


def _run_process(command, log, timeout):
    # Workers may spawn hipcc or another MIGraphX compiler process. Terminate
    # the whole owned process group on timeout/interruption before continuing.
    with subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    ) as process:
        try:
            return process.wait(timeout=timeout)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                return 124
            raise


def _worker(job, path, timeout=1800):
    write_json(path, job)
    with path.with_suffix(".log").open("a") as log:
        return _run_process(
            [sys.executable, "-m", "embeddinggemma2_rocm._compiler.worker", str(path)],
            log,
            timeout,
        )


def _probe(session):
    if _worker(dict(kind="probe"), session / "probe-job.json", 60):
        raise RuntimeError(
            "GPU health check failed; stop here and resume after recovering the device"
        )


def _measure(bundle, budget, batch, path, repeat, retry_failed=False):
    if path.exists():
        previous = _read_checkpoint(path, None)
        if previous is not None and previous.get("status") == "ok":
            try:
                saved = np.load(path.with_suffix(".npy"), allow_pickle=False)
                if saved.shape != (batch, 768) or not np.isfinite(saved).all():
                    previous = None
            except (OSError, ValueError, EOFError):
                previous = None
        if (
            previous is not None
            and (not retry_failed or previous.get("status") == "ok")
            and (previous.get("status") != "ok" or path.with_suffix(".npy").exists())
        ):
            return previous
        if path.exists():
            path.replace(path.with_suffix(".previous.json"))
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
        "--verify-reversal",
        "--unique-images",
    ]
    with path.with_suffix(".log").open("w") as log:
        code = _run_process(cmd, log, 1800)
    if code or not path.exists():
        record = _read_checkpoint(path, {})
        if record.get("status") in (None, "ok"):
            record["status"] = "process_error"
        record["returncode"] = code
        write_json(path, record)
    return json.loads(path.read_text())


def _screen(job, session):
    """A crashing candidate is recorded once; remaining candidates run in a fresh process."""
    path = Path(job["output"])
    while True:
        before = _read_checkpoint(path, [])
        # An interrupted launch is not assumed safe or successful on resume.
        for record in before:
            if record["status"] == "running":
                record.update(
                    status="failed", error="worker interrupted during this candidate"
                )
        write_json(path, before)
        if len(before) == len(job["candidates"]):
            break
        code = _worker(job, path.with_name("job.json"))
        after = json.loads(path.read_text())
        if code:
            _probe(session)
        if len(after) <= len(before):
            raise RuntimeError(
                f"Kernel screening made no progress: {path.parent}; inspect job.log and resume"
            )
    passed = [r for r in before if r.get("numeric_pass") and r["status"] == "ok"]
    return min(passed, key=lambda r: r["median_seconds"])["config"] if passed else None


def tune(a):
    from ._compiler.search import candidates, combinations
    from .presets import environment, profile_key

    if a.budget == 280 and a.batch > 32:
        raise ValueError("Budget 280 tuning supports request batches up to 32")
    if a.repeat < 1:
        raise ValueError("repeat must be positive")
    batches = sorted(
        set(v for v in a.vision_batches if 0 < v <= a.batch and a.batch % v == 0)
    )
    if not batches:
        raise ValueError("No vision batch divides the requested batch")
    output = Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = environment()
    identity = dict(
        schema=2,
        environment=env,
        profile=profile_key(a.budget, a.batch, (16, 9), "fast"),
        vision_batches=batches,
        repeat=a.repeat,
        model=str(Path(a.model).resolve()),
        input_size=[512, 288],
    )
    session = (
        Path(a.resume).resolve()
        if a.resume
        else output / "tuning" / str(time.time_ns())
    )
    session.mkdir(parents=True, exist_ok=True)
    manifest = session / "session.json"
    if a.resume:
        if not manifest.exists() or json.loads(manifest.read_text()) != identity:
            raise ValueError(
                "Resume requires identical GPU, software, sources, model, profile and search parameters"
            )
    else:
        write_json(manifest, identity)
    print(f"Tuning session: {session}", flush=True)
    exports = output / "exports"
    exported = export_images(a.model, exports, [a.budget], (16, 9))[0]
    grid = exported["grid"]
    roots = []
    attempts = []
    cache = [str(output), *[str(Path(p).resolve()) for p in a.cache_dir]]

    def compile_one(name, vb, config, precision="fast"):
        if precision == "fast":
            attempts.append(name)
        dest = session / name
        dest.mkdir(exist_ok=True)
        if not (dest / "exports").exists():
            (dest / "exports").symlink_to(exports, target_is_directory=True)
        status = dest / "compile-status.json"
        if status.exists():
            return dest if json.loads(status.read_text())["status"] == "ok" else None
        print(f"Compiling {name}", flush=True)
        job = dict(
            kind="compile",
            model=identity["model"],
            output=str(dest),
            budget=a.budget,
            batch=a.batch,
            aspect=[16, 9],
            precision=precision,
            vision_batch=vb,
            kernel_config=config,
            cache_dirs=[*cache, *[str(r["bundle"]) for r in roots]],
        )
        code = _worker(job, dest / "compile-job.json", 3600)
        write_json(
            status, dict(status="ok" if code == 0 else "failed", returncode=code)
        )
        if code:
            _probe(session)
        return dest if code == 0 else None

    reference = session / "reference-cpu.json"
    ref_result = _read_checkpoint(reference, None)
    if ref_result is None or not reference.with_suffix(".npy").exists():
        print("Computing independent CPU FP32 calibration", flush=True)
        with reference.with_suffix(".log").open("w") as log:
            code = _run_process(
                [
                    sys.executable,
                    "-m",
                    "embeddinggemma2_rocm.reference",
                    "--model",
                    identity["model"],
                    "--exports",
                    str(exports),
                    "--budget",
                    str(a.budget),
                    "--batch",
                    str(a.batch),
                    "--output",
                    str(reference),
                ],
                log,
                3600,
            )
            if code:
                raise RuntimeError(
                    f"CPU FP32 worker failed ({code}); inspect reference-cpu.log and resume"
                )
        ref_result = json.loads(reference.read_text())
    if ref_result.get("status") != "ok":
        raise RuntimeError("CPU FP32 calibration failed; no profile can be selected")
    ref = np.load(reference.with_suffix(".npy"))
    for vb in batches:
        baseline = compile_one(f"vision{vb}-default", vb, None)
        if baseline:
            roots.append(
                dict(
                    name=f"vision{vb}-default",
                    vision_batch=vb,
                    bundle=str(baseline),
                    kernel_config=None,
                )
            )
        best = {}
        for family in ("gelu", "qk", "norm", "preprocess"):
            print(f"Screening vision_batch={vb}: {family}", flush=True)
            dest = session / f"micro-v{vb}" / family
            dest.mkdir(parents=True, exist_ok=True)
            job = dict(
                kind="micro",
                family=family,
                vision_batch=a.batch if family == "preprocess" else vb,
                patches=grid[0] * grid[1],
                grid=grid,
                input_size=[512, 288],
                arch=env["device"]["arch"].split(":")[0],
                output=str(dest / "results.json"),
                candidates=candidates(family, dict(env["device"], **env["limits"])),
            )
            best[family] = _screen(job, session)
        for config in combinations(best):
            key = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()[:12]
            name = f"vision{vb}-{key}"
            dest = compile_one(name, vb, config)
            if dest:
                roots.append(
                    dict(
                        name=name,
                        vision_batch=vb,
                        bundle=str(dest),
                        kernel_config=config,
                    )
                )
    records = []
    for round_, order in enumerate((roots, roots[::-1])):
        for candidate in order:
            print(f"Measuring round {round_}: {candidate['name']}", flush=True)
            path = session / f"{candidate['name']}-round{round_}.json"
            result = _measure(candidate["bundle"], a.budget, a.batch, path, a.repeat)
            result.update(name=candidate["name"], round=round_)
            if result.get("status") == "ok":
                x = np.load(path.with_suffix(".npy"))
                cos = (x * ref).sum(1) / (
                    np.linalg.norm(x, axis=1) * np.linalg.norm(ref, axis=1)
                )
                result["min_cosine_to_standard"] = float(cos.min())
                result["numeric_pass"] = bool(
                    np.isfinite(cos).all()
                    and cos.min() >= 0.995
                    and result["input_sha256"] == ref_result["input_sha256"]
                )
            elif result.get("status") == "process_error":
                _probe(session)
            records.append(result)
            write_json(session / "measurements.json", records)
    qualified = []
    for candidate in roots:
        group = [r for r in records if r["name"] == candidate["name"]]
        if len(group) == 2 and all(r.get("numeric_pass") for r in group):
            qualified.append(
                dict(
                    candidate,
                    ms_per_image=float(
                        np.median([t for r in group for t in r["seconds"]])
                        * 1000
                        / a.batch
                    ),
                    min_cosine_to_standard=min(
                        r["min_cosine_to_standard"] for r in group
                    ),
                )
            )
    best = min(qualified, key=lambda r: r["ms_per_image"]) if qualified else None
    screening = {}
    for vb in batches:
        screening[str(vb)] = {}
        for family in ("gelu", "qk", "norm", "preprocess"):
            checked = json.loads(
                (session / f"micro-v{vb}" / family / "results.json").read_text()
            )
            screening[str(vb)][family] = dict(
                attempted=len(checked),
                passed=sum(
                    r.get("numeric_pass", False) and r["status"] == "ok"
                    for r in checked
                ),
            )
    report = dict(
        schema=2,
        search=dict(
            vision_batches=batches,
            kernels=screening,
            pipeline_attempted=len(attempts),
            pipeline_compiled=len(roots),
        ),
        complete=True,
        environment=env,
        profile=identity["profile"],
        budget=a.budget,
        batch=a.batch,
        selected=best,
        candidates=qualified,
        measurements=records,
        reference_backend=ref_result["backend"],
        input_sha256=ref_result["input_sha256"],
        activated=False,
        quality_status="Retrieval validation required before publishing a recipe",
    )
    write_json(session / "profile.json", report)
    print(
        json.dumps(
            dict(profile=str(session / "profile.json"), selected=best), indent=2
        ),
        flush=True,
    )
    if best is None:
        raise RuntimeError(
            "No candidate passed both measurement rounds and numerical validation"
        )
    return report
