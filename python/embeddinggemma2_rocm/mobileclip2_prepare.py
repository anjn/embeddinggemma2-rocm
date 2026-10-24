"""Offline MobileCLIP2 export/compile. Heavy dependencies are preparation-only."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REVISION = "ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6"
REPO = "timm/MobileCLIP2-S2-OpenCLIP"


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def export_tokenizer(root, tokenizer):
    import html

    import ftfy
    from tokenizers import (
        AddedToken,
        Regex,
        Tokenizer,
        models,
        normalizers,
        pre_tokenizers,
        processors,
    )

    # Export the exact SimpleTokenizer vocabulary/merges, not a different HF tokenizer.
    bpe = models.BPE(
        vocab=tokenizer.encoder,
        merges=list(tokenizer.bpe_ranks),
        end_of_word_suffix="</w>",
    )
    tok = Tokenizer(bpe)
    tok.add_special_tokens(
        [
            AddedToken("<start_of_text>", normalized=True, special=True),
            AddedToken("<end_of_text>", normalized=True, special=True),
        ]
    )
    # Portable character cleanup for ordinary UTF-8; mojibake repair is excluded.
    replacements = []
    for code in [
        *range(0xFF00, 0xFFF0),
        *range(0xFB00, 0xFB07),
        *range(0x2018, 0x2020),
    ]:
        char = chr(code)
        fixed = ftfy.fix_text(char)
        if fixed != char:
            replacements.append(normalizers.Replace(char, fixed))
    tok.normalizer = normalizers.Sequence(
        [
            *replacements,
            normalizers.NFC(),
            normalizers.Replace(Regex(r"\s+"), " "),
            normalizers.Strip(),
            normalizers.Lowercase(),
        ]
    )
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(tokenizer.pat.pattern), behavior="removed", invert=True
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.post_processor = processors.TemplateProcessing(
        single="<start_of_text> $A <end_of_text>",
        special_tokens=[("<start_of_text>", 49406), ("<end_of_text>", 49407)],
    )
    tok.enable_truncation(max_length=77)
    tok.enable_padding(length=77, pad_id=0, pad_token="!")
    tok.save(str(root / "tokenizer.json"))
    cases = [
        "",
        "a person wearing a hat",
        "日本語の検索",
        "It's a RED shirt!",
        "café\tblue\nshirt",
        "person " * 100,
        "<start_of_text>hello<end_of_text>",
        "<START_OF_TEXT>Hello",
        "a “blue” shirt",
        "a ﬂoral shirt",
        "Ａ ＢＬＵＥ shirt",
        "a &amp;amp; b",
        "e\u0301 cafe",
        "white-shirt 123",
        "\u3000日本語\u00a0の服",
    ]
    for text in cases:
        actual = tok.encode(html.unescape(html.unescape(text))).ids
        expected = tokenizer([text])[0].tolist()
        if actual != expected:
            raise RuntimeError(
                f"Tokenizer export differs for {text!r}: {actual[:15]} vs {expected[:15]}"
            )

    write_json(
        root / "tokenizer-cases.json",
        [{"text": text, "ids": tokenizer([text])[0].tolist()} for text in cases],
    )
    return cases


def export(root):
    import numpy as np
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download
    from timm.utils import reparameterize_model

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint = hf_hub_download(REPO, "open_clip_model.safetensors", revision=REVISION)
    model = open_clip.create_model("MobileCLIP2-S2", pretrained=checkpoint).eval()
    model = reparameterize_model(model, inplace=True)
    tokenizer = open_clip.get_tokenizer("MobileCLIP2-S2")
    cases = export_tokenizer(root, tokenizer)

    class ImageEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = model.visual

        def forward(self, images):
            return torch.nn.functional.normalize(self.visual(images), dim=-1)

    class TextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text = model.text

        def forward(self, tokens):
            return torch.nn.functional.normalize(self.text(tokens), dim=-1)

    with torch.inference_mode():
        torch.onnx.export(
            ImageEncoder().eval(),
            torch.zeros(1, 3, 256, 256),
            str(root / "image.onnx"),
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        tokens = tokenizer(cases)
        torch.onnx.export(
            TextEncoder().eval(),
            tokens[:1],
            str(root / "text.onnx"),
            input_names=["tokens"],
            output_names=["embeddings"],
            opset_version=17,
            dynamo=False,
        )
        np.save(
            root / "text-reference.npy",
            model.encode_text(tokens, normalize=True).numpy(),
        )
        write_json(root / "text-cases.json", cases)
    write_json(
        root / "manifest.json",
        {
            "format": 1,
            "model": "MobileCLIP2-S2",
            "dimension": 512,
            "revision": REVISION,
            "checkpoint_sha256": digest(checkpoint),
            "preprocessing": "letterbox-bilinear-256",
            "image_sha256": digest(root / "image.onnx"),
            "text_sha256": digest(root / "text.onnx"),
            "tokenizer_sha256": digest(root / "tokenizer.json"),
        },
    )


def seal_backend(root, backend):
    """Record the complete compiled artifact set; native loading verifies hashes."""
    meta = json.loads((root / f"{backend}.json").read_text())
    if backend == "gpu":
        files = [root / f"gpu-b{b}.mxr" for b in (1, 4)]
    else:
        files = [root / "npu-config.json"] + sorted(
            p
            for p in (root / "npu-cache" / meta["cache_key"]).rglob("*")
            if p.is_file()
        )
    meta["artifacts"] = {str(p.relative_to(root)): digest(p) for p in files}
    meta["machine"] = platform.machine()
    meta["system"] = platform.system()
    write_json(root / f"{backend}.json", meta)


def compile_backend(root, backend):
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(root / "image.onnx") != manifest["image_sha256"]:
        raise ValueError("Image ONNX checksum mismatch")
    if backend == "gpu":
        disabled = set(
            filter(None, os.environ.get("MIGRAPHX_DISABLE_PASSES", "").split(","))
        )
        disabled.add("simplify_algebra")
        os.environ["MIGRAPHX_DISABLE_PASSES"] = ",".join(sorted(disabled))
        os.environ.setdefault("MIGRAPHX_GPU_COMPILE_PARALLEL", "4")
        import migraphx as mg
        import numpy as np

        for batch in (1, 4):
            p = mg.parse_onnx(
                str(root / "image.onnx"),
                map_input_dims={"images": [batch, 3, 256, 256]},
            )
            mg.quantize_fp16(p)
            p.compile(mg.get_target("gpu"), offload_copy=True)
            mg.save(p, str(root / f"gpu-b{batch}.mxr"))
            pixels = np.zeros((batch, 3, 256, 256), np.float32)
            result = np.array(p.run({"images": mg.argument(pixels)})[0])
            if result.shape != (batch, 512) or not np.isfinite(result).all():
                raise RuntimeError("GPU validation failed")
        write_json(
            root / "gpu.json",
            {
                "image_sha256": manifest["image_sha256"],
                "batches": [1, 4],
                "precision": "fp16",
                "disable_passes": os.environ["MIGRAPHX_DISABLE_PASSES"],
                "migraphx": getattr(mg, "__version__", "unknown"),
            },
        )
    else:
        import numpy as np
        import onnxruntime as ort

        config = {
            "passes": [
                {"name": "init", "plugin": "vaip-pass_init"},
                {
                    "name": "vaiml_partition",
                    "plugin": "vaip-pass_vaiml_partition",
                    "vaiml_config": {
                        "optimize_level": 3,
                        "preferred_data_storage": "unvectorized",
                        "tp_size": 2,
                    },
                },
            ],
            "target": "VAIML",
            "targets": [{"name": "VAIML", "pass": ["init", "vaiml_partition"]}],
        }
        write_json(root / "npu-config.json", config)
        key = hashlib.sha256(
            (
                manifest["image_sha256"]
                + ort.__version__
                + json.dumps(config, sort_keys=True)
            ).encode()
        ).hexdigest()[:24]
        work = root / "npu-work"
        work.mkdir(exist_ok=True)
        os.chdir(work)
        o = ort.SessionOptions()
        o.intra_op_num_threads = 4
        o.add_free_dimension_override_by_name("batch", 1)
        o.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        opts = {
            "config_file": str(root / "npu-config.json"),
            "cache_dir": str(root / "npu-cache"),
            "cache_key": key,
            "enable_cache_file_io_in_mem": "0",
        }
        s = ort.InferenceSession(
            str(root / "image.onnx"),
            sess_options=o,
            providers=["VitisAIExecutionProvider"],
            provider_options=[opts],
        )
        s.disable_fallback()
        y = s.run(None, {"images": np.zeros((1, 3, 256, 256), np.float32)})[0]
        if y.shape != (1, 512) or not np.isfinite(y).all():
            raise RuntimeError("NPU validation failed")
        write_json(
            root / "npu.json",
            {
                "image_sha256": manifest["image_sha256"],
                "batch": 1,
                "precision": "bf16",
                "onnxruntime": ort.__version__,
                "cache_key": key,
            },
        )

    seal_backend(root, backend)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["export", "gpu", "npu"])
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    root = a.bundle.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.action == "export" or a.worker:
        with (root / ".prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if a.action == "export":
                export(root)
            else:
                compile_backend(root, a.action)
    else:
        # Backend settings and compiler CWD never affect the caller's process.
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                a.action,
                "--bundle",
                str(root),
                "--worker",
            ],
            check=True,
            cwd=root,
        )


if __name__ == "__main__":
    main()
