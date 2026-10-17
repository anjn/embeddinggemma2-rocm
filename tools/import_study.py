"""Migrate locally generated study artifacts; never commit the resulting model bundle."""

import argparse, json, shutil, hashlib
from pathlib import Path
import numpy as np


def tensor(root, name, value):
    value = np.ascontiguousarray(value)
    path = root / (name + ".bin")
    value.tofile(path)
    return dict(
        file=path.name,
        dtype="bfloat16" if value.dtype == np.uint16 else str(value.dtype),
        lens=list(value.shape),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def migrate(study, dest, batches):
    study = Path(study).resolve()
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    profiles = []
    for batch in batches:
        if batch not in (1, 4, 16, 32, 64, 128):
            raise ValueError("Unsupported migration batch")
        vb = min(batch, 16)
        bb = batch if batch in (1, 4, 16, 64, 128) else 16
        root = dest / f"budget70-b{batch}"
        root.mkdir(exist_ok=True)
        inp = study / f"artifacts/best-bf16-batches/budget-70-b{bb}/full-vision-bf16"
        data = dict(np.load(inp / "backbone-inputs.npz"))
        tensors = {}
        for name, key in [("template", "inputs_embeds"), ("mask", "attention_mask")]:
            tensors[name] = tensor(
                root,
                name,
                np.tile(
                    data[key],
                    (batch // bb, 1, 1) if key == "inputs_embeds" else (batch // bb, 1),
                ),
            )
        shutil.copy2(
            study / f"artifacts/hip-device/b{bb}/backbone.mxr", root / "backbone.mxr"
        )
        if vb == 16:
            source = study / "artifacts/hip-device/b16/sdpa-ffn-norm-segments"
            target = root / "vision"
            target.mkdir(exist_ok=True)
            m = json.loads((source / "manifest.json").read_text())
            m["weight_files"] = {}
            for stage in m["stages"]:
                shutil.copy2(source / stage["path"], target / stage["path"])
            with np.load(source / "weights.npz") as weights:
                for name in weights.files:
                    m["weight_files"][name] = tensor(target, name, weights[name])
            # Keep portable provenance, not private absolute paths.
            m["source"] = Path(m["source"]).name
            (target / "manifest.json").write_text(json.dumps(m, indent=2))
            vision = dict(
                kind="segmented", path="vision", batch=vb, grid=[18, 33], tokens=66
            )
        else:
            source = study / f"artifacts/hip-device/b{vb}/vision.mxr"
            shutil.copy2(source, root / "vision.mxr")
            vision = dict(
                kind="compiled", path="vision.mxr", batch=vb, grid=[18, 33], tokens=66
            )
        bundle = dict(
            schema=1,
            model="google/embeddinggemma-2",
            revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
            gpu_arch="gfx1150",
            budget=70,
            batch=batch,
            aspect=[16, 9],
            vision=vision,
            backbone=dict(path="backbone.mxr", batch=bb),
            tensors=tensors,
        )
        (root / "bundle.json").write_text(json.dumps(bundle, indent=2))
        profiles.append(dict(path=root.name, budget=70, batch=batch, aspect=[16, 9]))
    shutil.copy2(study / "models/google/tokenizer.json", dest / "tokenizer.json")
    (dest / "index.json").write_text(
        json.dumps(
            dict(
                schema=1,
                model="google/embeddinggemma-2",
                tokenizer="tokenizer.json",
                profiles=profiles,
            ),
            indent=2,
        )
    )
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 32, 128])
    a = p.parse_args()
    print(migrate(a.study, a.output, a.batches))
