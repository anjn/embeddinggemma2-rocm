"""Offline model export and per-GPU compilation. Runtime does not import this module."""

from __future__ import annotations
import gc, hashlib, json, os, platform, shutil, subprocess, sys, tempfile, time
from math import gcd
from pathlib import Path
import numpy as np

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
GGUF_ID = "ggml-org/embeddinggemma-2-GGUF"
GGUF_REVISION = "bfcd298762cc34d0357ece5ebdd31791a3a374d8"


def cache_root():
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "embeddinggemma2-rocm"
    )


def fetch_model(directory=None):
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=directory)
    )


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        name = f.name
    os.replace(name, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_index(root):
    p = Path(root) / "index.json"
    return (
        json.loads(p.read_text())
        if p.exists()
        else dict(
            schema=1,
            model=MODEL_ID,
            revision=MODEL_REVISION,
            profiles=[],
            text_profiles=[],
        )
    )


def save_tensor(root, name, array):
    a = np.ascontiguousarray(array)
    p = Path(root) / (name + ".bin")
    a.tofile(p)
    return dict(
        file=p.name,
        dtype="bfloat16" if a.dtype == np.uint16 else str(a.dtype),
        lens=list(a.shape),
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    )


def load_model(model):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    st = SentenceTransformer(
        str(model),
        device="cpu",
        model_kwargs={
            "dtype": torch.float32,
            "attn_implementation": {
                "": "eager",
                "text_config": "eager",
                "vision_config": "eager",
                "audio_config": "eager",
            },
        },
        config_kwargs={"audio_config": None},
    )
    return st.eval()


def wrappers():
    import torch
    from torch import nn

    class Backbone(nn.Module):
        def __init__(self, lm, upstream_mask=False):
            super().__init__()
            self.lm = lm
            self.upstream_mask = upstream_mask

        def forward(self, inputs_embeds, attention_mask):
            n = inputs_embeds.shape[1]
            pos = torch.arange(n, device=inputs_embeds.device)
            valid = attention_mask[:, None, None, :].bool()
            local = valid & (
                (pos[:, None] - pos[None, :]).abs() <= self.lm.config.sliding_window
            )
            if self.upstream_mask:
                full = torch.zeros_like(valid, dtype=inputs_embeds.dtype).masked_fill(
                    ~valid, torch.finfo(inputs_embeds.dtype).min
                )
                local = torch.zeros_like(local, dtype=inputs_embeds.dtype).masked_fill(
                    ~local, torch.finfo(inputs_embeds.dtype).min
                )
            else:
                full = (
                    1 - attention_mask[:, None, None, :].to(inputs_embeds.dtype)
                ) * (-10000.0)
                local = (1 - local.to(inputs_embeds.dtype)) * (-10000.0)
            hidden = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask={"full_attention": full, "sliding_attention": local},
                return_dict=True,
            ).last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, dim=-1)

    class Text(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.embedding = model.get_input_embeddings()
            self.backbone = Backbone(model.language_model)

        def forward(self, input_ids, attention_mask):
            return self.backbone(self.embedding(input_ids), attention_mask)

    class Vision(nn.Module):
        def __init__(self, model, positions):
            super().__init__()
            self.tower = model.vision_tower
            self.project = model.embed_vision
            self.register_buffer("positions", positions)
            self.register_buffer(
                "padding", torch.zeros(positions.shape[:2], dtype=torch.bool)
            )
            xy = positions[0].tolist()
            gw = max(x for x, y in xy) + 1
            groups = {}
            for i, (x, y) in enumerate(xy):
                groups.setdefault((y // 3) * (gw // 3) + x // 3, []).append(i)
            assert all(len(v) == 9 for v in groups.values())
            self.register_buffer(
                "indices",
                torch.tensor([groups[k] for k in sorted(groups)], dtype=torch.long),
            )

        def forward(self, pixel_values):
            positions = self.positions.expand(pixel_values.shape[0], -1, -1)
            padding = self.padding.expand(pixel_values.shape[0], -1)
            x = self.tower.patch_embedder(pixel_values, positions, padding)
            rope = self.tower.encoder.rotary_emb(x, positions)
            for layer in self.tower.encoder.layers:
                x = layer(
                    x,
                    attention_mask=None,
                    position_embeddings=rope,
                    position_ids=positions,
                )
            pooled = (
                x[:, self.indices, :].float().mean(2)
                * self.tower.config.hidden_size**0.5
            )
            return (
                self.project(pooled.to(self.project.embedding_projection.weight.dtype))
                .float()
                .reshape(-1, 512)
            )

    return Backbone, Text, Vision


def normalized_aspect(aspect):
    width, height = aspect
    if width < 1 or height < 1:
        raise ValueError("Aspect dimensions must be positive")
    factor = gcd(width, height)
    return width // factor, height // factor


def fixture_size(aspect):
    width, height = normalized_aspect(aspect)
    # Preserve exact large coprime ratios without scaling an HD image by 32.
    scale = min(32, max(1, 1024 // max(width, height)))
    return width * scale, height * scale


def export_images(model, root, budgets, aspect):
    import torch, onnx
    from PIL import Image

    aspect = normalized_aspect(aspect)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    Backbone, _, Vision = wrappers()
    st = None
    exports = []
    for budget in budgets:
        dest = root / f"budget{budget}-aspect{aspect[0]}x{aspect[1]}"
        if (dest / "export.json").exists() and not (
            budget == 70
            and list(aspect) == [16, 9]
            and not (dest / "vision-fast-v3-b16.onnx").exists()
        ):
            exports.append(json.loads((dest / "export.json").read_text()))
            continue
        if st is None:
            st = load_model(model)
        dest.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", fixture_size(aspect), (127, 95, 63))
        features = st[0].preprocess(
            [{"image": image}],
            prompt="",
            processing_kwargs={"image": {"max_soft_tokens": budget}},
        )
        saved = {}

        def capture(module, args, kwargs):
            saved.update(
                {
                    k: v.detach().clone()
                    for k, v in kwargs.items()
                    if isinstance(v, torch.Tensor)
                }
            )

        hook = st[0].model.language_model.register_forward_pre_hook(
            capture, with_kwargs=True
        )
        try:
            with torch.inference_mode():
                reference = st(features)["sentence_embedding"].float()
        finally:
            hook.remove()
        valid = (features["image_position_ids"][0] >= 0).all(-1)
        positions = features["image_position_ids"][:, valid]
        pixels = features["pixel_values"][:, valid]
        grid = [int(positions[..., 1].max()) + 1, int(positions[..., 0].max()) + 1]
        tokens = int(valid.sum()) // 9
        slots = features["input_ids"] == st[0].model.config.image_token_id
        assert (
            int(slots.sum()) == tokens and saved["inputs_embeds"].shape[1] == tokens + 4
        )
        np.savez(
            dest / "template.npz",
            inputs_embeds=saved["inputs_embeds"].numpy(),
            attention_mask=features["attention_mask"].numpy(),
        )
        np.save(dest / "reference.npy", reference.numpy())
        np.save(dest / "pixels.npy", pixels.numpy())
        with torch.inference_mode():
            vision = Vision(st[0].model, positions).eval().requires_grad_(False)
            torch.onnx.export(
                vision,
                (pixels,),
                str(dest / "vision.onnx"),
                dynamo=False,
                opset_version=18,
                input_names=["pixel_values"],
                output_names=["output"],
                dynamic_axes={
                    "pixel_values": {0: "batch"},
                    "output": {0: "media_tokens"},
                },
                external_data=True,
            )
            backbone = Backbone(st[0].model.language_model).eval().requires_grad_(False)
            torch.onnx.export(
                backbone,
                (saved["inputs_embeds"], features["attention_mask"]),
                str(dest / "backbone.onnx"),
                dynamo=False,
                opset_version=18,
                input_names=["inputs_embeds", "attention_mask"],
                output_names=["output"],
                dynamic_axes={
                    "inputs_embeds": {0: "batch"},
                    "attention_mask": {0: "batch"},
                    "output": {0: "batch"},
                },
                external_data=True,
            )
        onnx.checker.check_model(str(dest / "vision.onnx"))
        onnx.checker.check_model(str(dest / "backbone.onnx"))
        if budget == 70 and list(aspect) == [16, 9]:
            from ._compiler.recipe import mixed_vision, static_vision

            mixed_vision(st[0].model)
            with torch.inference_mode():
                fast = static_vision(st[0].model, positions, 16)
                torch.onnx.export(
                    fast,
                    (pixels.repeat(16, 1, 1),),
                    str(dest / "vision-fast-v3-b16.partial.onnx"),
                    dynamo=False,
                    opset_version=18,
                    input_names=["pixel_values"],
                    output_names=["output"],
                    external_data=True,
                )
            onnx.checker.check_model(str(dest / "vision-fast-v3-b16.partial.onnx"))
            os.replace(
                dest / "vision-fast-v3-b16.partial.onnx",
                dest / "vision-fast-v3-b16.onnx",
            )
            del fast, st
            st = None
            gc.collect()
        record = dict(
            schema=1,
            budget=budget,
            aspect=list(aspect),
            grid=grid,
            tokens=tokens,
            directory=dest.name,
            model=MODEL_ID,
            revision=MODEL_REVISION,
        )
        write_json(dest / "export.json", record)
        exports.append(record)
        print("exported", record, flush=True)
        del features, saved, vision, backbone
        gc.collect()
    return exports


def compile_onnx(path, shapes, precision):
    import migraphx

    def parse():
        p = migraphx.parse_onnx(str(path), map_input_dims=shapes)
        if precision == "fast":
            migraphx.quantize_bf16(p, ["dot", "convolution"])
        return p

    p = parse()
    try:
        p.compile(migraphx.get_target("gpu"), offload_copy=False, fast_math=False)
    except RuntimeError as error:
        if (
            "No solutions" not in str(error)
            or os.environ.get("EG2_MLIR_FALLBACK") == "1"
        ):
            raise
        print(
            "MLIR has no supported kernel for this shape; compiling standard GPU operators",
            flush=True,
        )
        # MIGraphX caches environment options. A fresh process is required for
        # this fallback; changing os.environ after compilation is ineffective.
        del p
        gc.collect()
        env = dict(os.environ, MIGRAPHX_DISABLE_MLIR="1", EG2_MLIR_FALLBACK="1")
        env.pop("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", None)
        with tempfile.TemporaryDirectory(prefix="eg2-compile-") as directory:
            result = str(Path(directory) / "program.mxr")
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import json,sys,migraphx; "
                    "from embeddinggemma2_rocm.preparation import compile_onnx; "
                    "p=compile_onnx(sys.argv[1],json.loads(sys.argv[2]),sys.argv[3]); "
                    "migraphx.save(p,sys.argv[4])",
                    str(path),
                    json.dumps(shapes),
                    precision,
                    result,
                ],
                env=env,
                check=True,
            )
            p = migraphx.load(result)
    return p


def export_fixed_backbone(model, source, batch):
    """Export the specialized image backbone with fully static dimensions."""
    import torch
    import onnx

    source = Path(source)
    path = source / f"backbone-static-v3-b{batch}.onnx"
    if path.exists():
        return path
    st = load_model(model)
    Backbone, _, _ = wrappers()
    wrapper = (
        Backbone(st[0].model.language_model, upstream_mask=True)
        .eval()
        .requires_grad_(False)
    )
    with np.load(source / "template.npz") as data:
        embeds = torch.from_numpy(np.repeat(data["inputs_embeds"], batch, axis=0))
        mask = torch.from_numpy(np.repeat(data["attention_mask"], batch, axis=0))
    temporary = path.with_suffix(".partial.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (embeds, mask),
            str(temporary),
            dynamo=False,
            opset_version=18,
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["output"],
            external_data=True,
        )
    onnx.checker.check_model(str(temporary))
    os.replace(temporary, path)
    del st, wrapper
    gc.collect()
    return path


def export_custom_vision(model, source, batch):
    import torch, onnx
    from ._compiler.recipe import mixed_vision, static_vision

    source = Path(source)
    legacy = source / "vision-fast-v3-b16.onnx"
    if batch == 16 and legacy.exists():
        return legacy
    path = source / f"vision-custom-v4-b{batch}.onnx"
    if path.exists():
        return path
    export = json.loads((source / "export.json").read_text())
    gh, gw = export["grid"]
    y, x = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    positions = torch.stack((x, y), dim=-1).reshape(1, gh * gw, 2)
    pixels = torch.from_numpy(np.load(source / "pixels.npy")).repeat(batch, 1, 1)
    st = load_model(model)
    mixed_vision(st[0].model)
    with torch.inference_mode():
        fast = static_vision(st[0].model, positions, batch)
        temporary = path.with_suffix(".partial.onnx")
        torch.onnx.export(
            fast,
            (pixels,),
            str(temporary),
            dynamo=False,
            opset_version=18,
            input_names=["pixel_values"],
            output_names=["output"],
            external_data=True,
        )
        onnx.checker.check_model(str(temporary))
        os.replace(temporary, path)
    del st, fast
    gc.collect()
    return path


def compile_images(
    model,
    output,
    budgets,
    batches,
    aspect=(16, 9),
    precision="fast",
    vision_batch=None,
    cache_dirs=(),
    kernel_config=None,
    tuning_profile=None,
    use_presets=True,
):
    import torch
    import migraphx

    aspect = normalized_aspect(aspect)
    output = Path(output).resolve()
    index = read_index(output)
    if index.get("schema") != 1:
        raise ValueError(
            "Prepare profiles in a schema-1 source bundle, then convert with eg2 share"
        )
    exports = export_images(model, output / "exports", budgets, aspect)
    device = torch.cuda.get_device_properties(0)
    arch = device.gcnArchName.split(":")[0]
    base_fingerprint = dict(
        recipe_revision=2,
        gpu_arch=arch,
        device=device.name,
        rocm=torch.version.hip,
        torch=torch.__version__,
        migraphx=getattr(migraphx, "__version__", "unknown"),
        precision=precision,
        model_revision=MODEL_REVISION,
    )
    from .presets import environment, select, profile_key

    env = environment() if use_presets or tuning_profile else None
    for e in exports:
        source = output / "exports" / e["directory"]
        # The full 128-image/268-token compiled backbone crashed on gfx1150.
        # Larger requests also failed when chunked; the API rejects them.
        limit = 32 if e["budget"] == 280 else 128
        if e["budget"] == 280:
            index["profiles"] = [
                p
                for p in index["profiles"]
                if not (p["budget"] == 280 and p["batch"] > limit)
            ]
        execution_batches = sorted({min(batch, limit) for batch in batches})
        for batch in execution_batches:
            selected = (
                select(
                    profile_key(e["budget"], batch, aspect, precision),
                    env,
                    tuning_profile,
                )
                if env and kernel_config is None
                else None
            )
            recipe = kernel_config
            selected_vb = vision_batch
            if selected:
                if (
                    vision_batch is not None
                    and vision_batch != selected["vision_batch"]
                ):
                    if tuning_profile:
                        raise ValueError(
                            "Explicit vision batch conflicts with tuning recipe"
                        )
                    selected = None
                else:
                    recipe = selected["kernel_config"]
                    selected_vb = selected["vision_batch"]
            vb = min(batch, selected_vb or (4 if e["budget"] == 280 else 16))
            while batch % vb:
                vb -= 1
            specialized = (
                e["budget"] == 70
                and vb == 16
                and precision == "fast"
                and arch == "gfx1150"
                and list(aspect) == [16, 9]
            )
            revision = 4 if recipe is not None else (3 if specialized else 2)
            fingerprint = dict(base_fingerprint, recipe_revision=revision)
            if recipe is not None:
                from .presets import recipe_digest

                fingerprint.update(kernel_config=recipe, recipe_sha256=recipe_digest())
            label = f"{arch}-budget{e['budget']}-b{batch}-aspect{aspect[0]}x{aspect[1]}-{precision}-v{vb}-r{revision}"
            if recipe is not None:
                label += (
                    "-"
                    + hashlib.sha256(
                        json.dumps(recipe, sort_keys=True).encode()
                    ).hexdigest()[:12]
                )
            dest = output / label
            dest.mkdir(parents=True, exist_ok=True)
            old = dest / "fingerprint.json"
            if old.exists() and json.loads(old.read_text()) != fingerprint:
                raise RuntimeError(
                    f"Cache fingerprint mismatch: {dest}; use a fresh output directory"
                )
            write_json(old, fingerprint)
            for stage, shapes in [
                ("vision", {"pixel_values": [vb, e["grid"][0] * e["grid"][1], 768]}),
                (
                    "backbone",
                    {
                        "inputs_embeds": [batch, e["tokens"] + 4, 512],
                        "attention_mask": [batch, e["tokens"] + 4],
                    },
                ),
            ]:
                path = dest / (stage + ".mxr")
                if path.exists():
                    continue
                batch_pattern = "*" if stage == "vision" else str(batch)
                vision_pattern = str(vb) if stage == "vision" else "*"
                pattern = f"{arch}-budget{e['budget']}-b{batch_pattern}-aspect{aspect[0]}x{aspect[1]}-{precision}-v{vision_pattern}-r{revision}*/{stage}.mxr"
                for cache_dir in [output, *map(Path, cache_dirs)]:
                    for cached in sorted(cache_dir.glob(pattern)):
                        metadata = cached.parent / "fingerprint.json"
                        if (
                            cached != path
                            and metadata.exists()
                            and {
                                k: v
                                for k, v in json.loads(metadata.read_text()).items()
                                if k != "kernel_config"
                            }
                            == {
                                k: v
                                for k, v in fingerprint.items()
                                if k != "kernel_config"
                            }
                        ):
                            temporary = path.with_suffix(".partial.mxr")
                            shutil.copy2(cached, temporary)
                            os.replace(temporary, path)
                            print(
                                "reuse " + stage,
                                cached.parent.name,
                                "for",
                                label,
                                flush=True,
                            )
                            break
                    if path.exists():
                        break
                if path.exists():
                    continue
                print("compile", label, stage, flush=True)
                onnx_path = (
                    (
                        source / "vision-fast-v3-b16.onnx"
                        if stage == "vision"
                        else export_fixed_backbone(model, source, batch)
                    )
                    if specialized
                    else source / (stage + ".onnx")
                )
                if recipe is not None:
                    onnx_path = (
                        export_custom_vision(model, source, vb)
                        if stage == "vision"
                        else export_fixed_backbone(model, source, batch)
                    )
                p = compile_onnx(onnx_path, shapes, precision)
                temporary = path.with_suffix(".partial.mxr")
                migraphx.save(p, str(temporary))
                os.replace(temporary, path)
                del p
                gc.collect()
            with np.load(source / "template.npz") as data:
                tensors = dict(
                    template=save_tensor(
                        dest,
                        "template",
                        np.repeat(data["inputs_embeds"], batch, axis=0),
                    ),
                    mask=save_tensor(
                        dest, "mask", np.repeat(data["attention_mask"], batch, axis=0)
                    ),
                )
            vision_config = dict(
                kind="compiled",
                path="vision.mxr",
                batch=vb,
                grid=e["grid"],
                tokens=e["tokens"],
            )
            if recipe is not None:
                from ._compiler.recipe import optimize

                try:
                    result = optimize(
                        dest / "vision.mxr",
                        dest / "optimized",
                        arch,
                        batch=vb,
                        patches=e["grid"][0] * e["grid"][1],
                        config=recipe,
                    )
                    vision_config.update(
                        kind="segmented" if result.is_dir() else "compiled",
                        path=str(result.relative_to(dest)),
                    )
                except (
                    ValueError,
                    AssertionError,
                    RuntimeError,
                    __import__("subprocess").SubprocessError,
                ) as error:
                    if not selected or tuning_profile:
                        raise
                    print(
                        "Published recipe failed its graph/compiler guards; "
                        "retaining the compiled GPU path:",
                        error,
                        flush=True,
                    )
                    selected = None
                    recipe = None
            elif (
                e["budget"] == 70
                and vb == 16
                and precision == "fast"
                and arch == "gfx1150"
                and (source / "vision-fast-v3-b16.onnx").exists()
            ):
                from ._compiler.recipe import optimize

                try:
                    segments = optimize(dest / "vision.mxr", dest / "optimized", arch)
                    vision_config.update(
                        kind="segmented", path=str(segments.relative_to(dest))
                    )
                except (
                    ValueError,
                    AssertionError,
                    RuntimeError,
                    __import__("subprocess").CalledProcessError,
                ) as error:
                    print(
                        "Guarded optimization unavailable; retaining compiled GPU path:",
                        error,
                        flush=True,
                    )
            bundle = dict(
                schema=1,
                model=MODEL_ID,
                revision=MODEL_REVISION,
                gpu_arch=arch,
                budget=e["budget"],
                batch=batch,
                aspect=list(aspect),
                vision=vision_config,
                backbone=dict(path="backbone.mxr", batch=batch),
                tensors=tensors,
                fingerprint=fingerprint,
                preprocess=(recipe or {}).get("preprocess")
                or dict(block=256, vector=1),
                tuning_source="published"
                if selected
                else "explicit"
                if recipe is not None
                else "default",
            )
            write_json(dest / "bundle.json", bundle)
            entry = dict(
                path=label,
                budget=e["budget"],
                batch=batch,
                aspect=list(aspect),
                precision=precision,
            )
            index["profiles"] = [
                p
                for p in index["profiles"]
                if not (
                    p["batch"] == batch
                    and p["budget"] == e["budget"]
                    and p["aspect"] == list(aspect)
                )
            ] + [entry]
            write_json(output / "index.json", index)
    shutil.copy2(Path(model) / "tokenizer.json", output / "tokenizer.json")
    index["tokenizer"] = "tokenizer.json"
    write_json(output / "index.json", index)
    return output


def prepare_text(
    model,
    output,
    batches=(1, 4, 32, 128),
    lengths=(32, 128, 512, 2048, 8192),
    precision="standard",
):
    import torch

    output = Path(output).resolve()
    if read_index(output).get("schema") != 1:
        raise ValueError(
            "Prepare text in a schema-1 source bundle, then convert with eg2 share"
        )
    directory = output / "exports/text-native"
    directory.mkdir(parents=True, exist_ok=True)
    path = output / f"text-{precision}-v5.pt"
    if not path.exists():
        st = load_model(model)
        _, Text, _ = wrappers()
        wrapper = Text(st[0].model).eval().requires_grad_(False)
        if precision == "fast":
            wrapper.to(torch.bfloat16)
        features = st[0].preprocess(
            ["task: search result | query: native ROCm embeddings"], prompt=""
        )
        base_ids = features["input_ids"]
        base_mask = features["attention_mask"]
        ids = torch.zeros((1, 32), dtype=torch.int64)
        mask = torch.zeros_like(ids)
        ids[:, : base_ids.shape[1]] = base_ids
        mask[:, : base_mask.shape[1]] = base_mask
        with torch.inference_mode():
            ref = wrapper(ids, mask).cpu()
            wrapper = wrapper.cuda()
            gpu_ids = ids.cuda()
            gpu_mask = mask.cuda()
            traced = torch.jit.trace(
                wrapper,
                (gpu_ids, gpu_mask),
                strict=False,
                check_inputs=[(gpu_ids.repeat(2, 1), gpu_mask.repeat(2, 1))],
            )

            # Keep CUDA device constants relative to the caller's device guard.
            def portable(block):
                for node in block.nodes():
                    if (
                        node.kind() == "prim::Constant"
                        and str(node.output().type()) == "Device"
                        and node.hasAttribute("value")
                        and node.kindOf("value") == "s"
                        and node.s("value").startswith("cuda")
                    ):
                        node.s_("value", "cuda")
                    for inner in node.blocks():
                        portable(inner)

            for module in traced.modules():
                for method in module._c._method_names():
                    portable(module._c._get_method(method).graph)
            temporary = path.with_suffix(".partial.pt")
            traced.save(str(temporary))
        np.savez(
            directory / "reference.npz",
            ids=ids.numpy(),
            mask=mask.numpy(),
            embeddings=ref.numpy(),
        )
        os.replace(temporary, path)
        del st, wrapper, traced
        gc.collect()
    index = read_index(output)
    # Do not promote the earlier experimental MIGraphX text export.
    index["text_profiles"] = [
        p
        for p in index.get("text_profiles", [])
        if p.get("kind") == "torchscript" and p.get("precision") != precision
    ]
    for batch in batches:
        for length in lengths:
            if not 1 <= batch <= 128 or not 2 <= length <= 8192:
                raise ValueError("Invalid text profile size")
            # Eager attention uses quadratic workspace; larger requests split over safe profiles.
            if batch * length * length > 8192 * 8192:
                continue
            index["text_profiles"].append(
                dict(
                    path=path.name,
                    kind="torchscript",
                    batch=batch,
                    sequence=length,
                    precision=precision,
                )
            )
    shutil.copy2(Path(model) / "tokenizer.json", output / "tokenizer.json")
    index["tokenizer"] = "tokenizer.json"
    write_json(output / "index.json", index)
    print(
        "Prepared native LibTorch text profiles; no Python interpreter is needed at runtime",
        flush=True,
    )
    return output
