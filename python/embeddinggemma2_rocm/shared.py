"""Convert trusted compiled bundles to external weights and shared workspaces.

The source bundle is never modified. Only the pinned MIGraphX serialization is
accepted; transformations preserve code objects, shapes, aliases and ordering.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .preparation import write_json

ITEMSIZE = {
    "bool_type": 1,
    "int8_type": 1,
    "uint8_type": 1,
    "half_type": 2,
    "bf16_type": 2,
    "int16_type": 2,
    "float_type": 4,
    "int32_type": 4,
    "uint32_type": 4,
    "double_type": 8,
    "int64_type": 8,
    "uint64_type": 8,
}


def shape_bytes(shape):
    if shape.get("sub_shapes") or shape.get("dynamic_dimensions"):
        raise ValueError("Shared storage requires static non-tuple allocations")
    lens, strides = shape["lens"], shape["strides"]
    if any(x == 0 for x in lens):
        return 0
    return (1 + sum((n - 1) * s for n, s in zip(lens, strides))) * ITEMSIZE[
        shape["type"]
    ]


def aligned(n):
    return (n + 255) // 256 * 256


def externalize_program(source, target, root, store):
    import msgpack

    v = msgpack.unpackb(Path(source).read_bytes(), raw=False)
    if v.get("version") != 7:
        raise ValueError("Unsupported MIGraphX serialization; expected version 7")
    if set(v["modules"]) != {"main"}:
        raise ValueError(
            "Shared conversion currently requires a single compiled main module"
        )
    external = {}
    nodes = v["modules"]["main"]["nodes"]
    for node in nodes:
        name = node["name"]
        if name == "hip::hip_copy_literal":
            literal = node["operator"]["literal"]
            if len(literal["data"]) != 1 or not isinstance(literal["data"][0], bytes):
                raise ValueError("Unsupported literal representation")
            raw = literal["data"][0]
            shape = literal["shape"]
            if len(raw) != shape_bytes(shape):
                raise ValueError("Literal storage extent mismatch")
            identity = hashlib.sha256(
                json.dumps(shape, sort_keys=True).encode() + b"\0" + raw
            ).hexdigest()
            spec = dict(
                file=f"weights/{identity}.bin",
                bytes=len(raw),
                shape=shape,
                sha256=hashlib.sha256(raw).hexdigest(),
            )
            file = root / spec["file"]
            if identity not in store:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(raw)
                store[identity] = spec
            parameter = "eg2_weight_" + node["output"].replace("@", "")
            node.update(name="@param", operator={"parameter": parameter}, inputs=[])
            external[parameter] = dict(weight=identity)
        elif name == "hip::hip_allocate_memory":
            parameter = "eg2_workspace_" + node["output"].replace("@", "")
            node.update(name="@param", operator={"parameter": parameter}, inputs=[])
            external[parameter] = dict(bytes=shape_bytes(node["shape"]))
        elif name in ("hip::allocate", "hip::hip_copy_to_gpu", "hip::copy_to_gpu"):
            raise ValueError(f"Unmanaged allocation/copy in compiled program: {name}")
    # Every non-weight parameter gets storage; runtime may bind a view instead.
    total = 0
    for node in nodes:
        if node["name"] != "@param":
            continue
        name = node["operator"]["parameter"]
        if "weight" not in external.get(name, {}):
            size = shape_bytes(node["shape"])
            external[name] = dict(bytes=size)
            total += aligned(size)
    temporary = Path(target).with_suffix(".shared.partial")
    temporary.write_bytes(msgpack.packb(v, use_bin_type=True))
    os.replace(temporary, target)
    write_json(
        str(target) + ".memory.json",
        dict(schema=1, parameters=external, workspace_bytes=total),
    )
    return total


def share_bundle(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError(
            "Shared conversion requires a new directory outside the source bundle"
        )
    index = json.loads((source / "index.json").read_text())
    if index.get("schema") != 1:
        raise ValueError("Convert an original schema-1 bundle")

    # Hard links save disk space. Every rewritten file is atomically replaced.
    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return dst

    shutil.copytree(
        source,
        destination,
        copy_function=link_or_copy,
        ignore=shutil.ignore_patterns("exports", "tuning", "*.npy", "*.npz", "*.log"),
    )
    store, peak = {}, 0
    try:
        for profile in index["profiles"]:
            root = destination / profile["path"]
            bundle = json.loads((root / "bundle.json").read_text())
            if bundle["budget"] == 280 and bundle["batch"] > 32:
                raise ValueError(
                    "Remove unstable budget280 profiles above batch32 before conversion"
                )
            total = externalize_program(
                root / bundle["backbone"]["path"],
                root / bundle["backbone"]["path"],
                destination,
                store,
            )
            vision = bundle["vision"]
            if vision["kind"] == "compiled":
                total += externalize_program(
                    root / vision["path"], root / vision["path"], destination, store
                )
            else:
                vr = root / vision["path"]
                manifest = json.loads((vr / "manifest.json").read_text())
                total += sum(
                    aligned(shape_bytes(s)) for s in manifest["allocs"].values()
                )
                total += sum(
                    aligned(shape_bytes(s["shape"]))
                    for s in manifest["params"].values()
                    if s["parameter"] != "pixel_values"
                )
                for stage in manifest["stages"]:
                    total += externalize_program(
                        vr / stage["path"], vr / stage["path"], destination, store
                    )
            b, t = bundle["batch"], vision["tokens"]
            total += aligned(b * (t + 4) * 512 * 4) + aligned(
                b * (t + 4) * np.dtype(bundle["tensors"]["mask"]["dtype"]).itemsize
            )
            total += aligned(b * np.prod(vision["grid"]) * 768 * 4) + aligned(
                b * 768 * 4
            )
            bundle["schema"] = 2
            bundle["workspace_bytes"] = int(total)
            write_json(root / "bundle.json", bundle)
            peak = max(peak, int(total))
        for profile in index.get("text_profiles", []):
            if profile.get("kind") != "torchscript":
                raise ValueError(
                    "Shared text profiles require the supported TorchScript path"
                )
            b, l = profile["batch"], profile["sequence"]
            peak = max(peak, 2 * aligned(b * l * 8) + aligned(b * 768 * 4))
        import torch

        text_weights = {}
        for path in sorted({p["path"] for p in index.get("text_profiles", [])}):
            module = torch.jit.load(str(source / path), map_location="cpu")
            groups = {}
            for group, items in [
                ("parameters", module.named_parameters(remove_duplicate=False)),
                ("buffers", module.named_buffers(remove_duplicate=False)),
            ]:
                entries = {}
                for name, t in items:
                    value = t.detach().contiguous()
                    types = {
                        torch.float32: "float_type",
                        torch.bfloat16: "bf16_type",
                        torch.float16: "half_type",
                        torch.int64: "int64_type",
                        torch.int32: "int32_type",
                        torch.bool: "bool_type",
                    }
                    shape = dict(
                        type=types[value.dtype],
                        sub_shapes=[],
                        lens=list(value.shape),
                        strides=list(value.stride()),
                        dynamic_dimensions=[],
                    )
                    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
                    identity = hashlib.sha256(
                        json.dumps(shape, sort_keys=True).encode() + b"\0" + raw
                    ).hexdigest()
                    if identity not in store:
                        spec = dict(
                            file=f"weights/{identity}.bin",
                            bytes=len(raw),
                            shape=shape,
                            sha256=hashlib.sha256(raw).hexdigest(),
                        )
                        (destination / spec["file"]).parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        (destination / spec["file"]).write_bytes(raw)
                        store[identity] = spec
                    entries[name] = dict(weight=identity)
                groups[group] = entries
            text_weights[path] = groups
            del module
        index.update(
            schema=2,
            shared=dict(
                schema=1, workspace_bytes=peak, weights=store, text_weights=text_weights
            ),
        )
        write_json(destination / "index.json", index)
    except Exception:
        # Mark incomplete output; never let it look like a usable legacy bundle.
        write_json(
            destination / "index.json",
            dict(schema=0, error="Shared conversion incomplete"),
        )
        raise
    return destination
