"""Independent CPU FP32 calibration with the runtime's bilinear/patch layout."""

from __future__ import annotations
import argparse
import hashlib
import os
import tempfile
from pathlib import Path
import numpy as np
from .benchmark import images
from .preparation import load_model, wrappers, write_json


def run(model, exports, budget, batch, output):
    import json
    import torch

    torch.set_num_threads(4)
    source = Path(exports) / f"budget{budget}-aspect16x9"
    gh, gw = json.loads((source / "export.json").read_text())["grid"]
    st = load_model(model)
    Backbone, _, Vision = wrappers()
    yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    positions = torch.stack((xx, yy), dim=-1).reshape(1, gh * gw, 2)
    vision = Vision(st[0].model, positions).eval().requires_grad_(False)
    backbone = (
        Backbone(st[0].model.language_model, upstream_mask=True)
        .eval()
        .requires_grad_(False)
    )
    with np.load(source / "template.npz") as template:
        base = torch.from_numpy(template["inputs_embeds"].copy())
        mask = torch.from_numpy(template["attention_mask"].copy())
    rgb = images(batch, unique=True)
    outputs = []
    with torch.inference_mode():
        for image in rgb:
            pixels = (
                torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float()
            )
            pixels = (
                torch.nn.functional.interpolate(
                    pixels,
                    size=(gh * 16, gw * 16),
                    mode="bilinear",
                    align_corners=False,
                )
                / 255
            )
            patches = (
                pixels.permute(0, 2, 3, 1)
                .reshape(1, gh, 16, gw, 16, 3)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(1, gh * gw, 768)
            )
            embedding = base.clone()
            media = vision(patches).reshape(1, -1, 512)
            embedding[:, 2 : 2 + media.shape[1]] = media
            outputs.append(backbone(embedding, mask).numpy().copy())
    result = np.concatenate(outputs)
    if result.shape != (batch, 768) or not np.isfinite(result).all():
        raise ValueError("Invalid CPU FP32 calibration output")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        np.save(f, result)
        f.flush()
        os.fsync(f.fileno())
        name = f.name
    os.replace(name, path.with_suffix(".npy"))
    write_json(
        path,
        dict(
            status="ok",
            backend="cpu_fp32_native_bilinear",
            budget=budget,
            batch=batch,
            input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "exports", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    run(args.model, args.exports, args.budget, args.batch, args.output)
