"""Direct-call CPU RGB to CPU embedding timing. One backend per process."""

from __future__ import annotations
import hashlib, json, platform, time
from pathlib import Path
import numpy as np


def images(batch):
    rng = np.random.default_rng(812)
    base = rng.integers(0, 256, (16, 288, 512, 3), dtype=np.uint8)
    return base[np.arange(batch) % 16].copy()


class TorchBackend:
    def __init__(self, path, budget, attention, precision="bf16"):
        import torch
        from sentence_transformers import SentenceTransformer
        from .preparation import MODEL_REVISION

        self.torch = torch
        self.precision = precision
        torch.set_num_threads(4)
        self.model = SentenceTransformer(
            str(path),
            device="cuda",
            model_kwargs={
                "dtype": torch.bfloat16 if precision == "bf16" else torch.float32,
                "attn_implementation": {
                    "": "eager",
                    "text_config": "eager",
                    "vision_config": attention,
                    "audio_config": "eager",
                },
            },
            config_kwargs={"audio_config": None},
        ).eval()
        self.budget = budget
        self.metadata = dict(
            torch=torch.__version__,
            rocm=torch.version.hip,
            gpu=torch.cuda.get_device_name(0),
            dtype=precision,
            parameter_dtype="bf16" if precision == "bf16" else "fp32",
            autocast_dtype="bf16" if precision == "mixed" else None,
            attention=attention,
            text_attention="eager",
            model_revision=MODEL_REVISION,
        )

    def encode(self, rgb):
        from PIL import Image
        from sentence_transformers.util import batch_to_device

        torch = self.torch
        with torch.inference_mode():
            data = self.model[0].preprocess(
                [{"image": Image.fromarray(im)} for im in rgb],
                prompt="",
                processing_kwargs={"image": {"max_soft_tokens": self.budget}},
            )
            self.metadata["actual_sequence_length"] = int(data["input_ids"].shape[1])
            self.metadata["actual_image_tokens"] = int(
                (
                    data["input_ids"][0] == self.model[0].model.config.image_token_id
                ).sum()
            )
            data = batch_to_device(data, "cuda")
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.precision == "mixed"
            ):
                out = self.model(data)["sentence_embedding"]
            out = torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()
        return out


def run(a):
    rgb = images(a.batch)
    import PIL

    record = dict(
        schema=1,
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            numpy=np.__version__,
            pillow=PIL.__version__,
        ),
        backend=a.backend,
        budget=a.budget,
        batch=a.batch,
        input_size=[512, 288],
        input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        scope="Direct in-process CPU uint8 RGB to CPU normalized float32 embeddings; includes preprocessing and transfers; excludes fixture generation/prescale, file IO, load, compile and warmup",
        warmup=a.warmup,
        repeat=a.repeat,
    )
    try:
        if a.backend == "native":
            from .runtime import Embedder

            model = Embedder(a.bundle)
            encode = lambda: model.encode_images(rgb, token_budget=a.budget)
            metadata = {"profiles": model.capabilities}
        elif a.backend == "torch":
            if not a.model:
                raise ValueError("--model is required for torch")
            model = TorchBackend(a.model, a.budget, a.attention, a.torch_precision)
            encode = lambda: model.encode(rgb)
            metadata = model.metadata
        else:
            from .llama import LlamaBackend

            model = LlamaBackend(
                a.llama_library,
                a.model,
                a.mmproj,
                a.budget,
                a.batch,
                a.flash == "on",
                a.vision_batch,
            )
            encode = lambda: model.encode(rgb)
            metadata = model.metadata
        for _ in range(a.warmup):
            out = encode()
        times = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            out = encode()
            times.append(time.perf_counter() - t)
        assert (
            out.shape == (a.batch, 768)
            and np.isfinite(out).all()
            and np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-3)
        )
        record.update(
            status="ok",
            seconds=times,
            median_ms_per_image=float(np.median(times) * 1000 / a.batch),
            images_per_second=float(a.batch / np.median(times)),
            metadata=metadata,
        )
        target = Path(a.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target.with_suffix(".npy"), out)
        if a.backend == "native":
            record["metadata"] = {"capabilities": model.capabilities}
        if hasattr(model, "close"):
            model.close()
    except Exception as e:
        from .runtime import UnsupportedProfile

        status = (
            "unsupported"
            if isinstance(e, NotImplementedError)
            else "unprepared"
            if isinstance(e, UnsupportedProfile)
            else ("out_of_memory" if "out of memory" in str(e).lower() else "error")
        )
        record.update(
            status=status,
            error=f"{type(e).__name__}: {e}",
            metadata=locals().get("metadata", {}),
        )
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    if record["status"] == "error":
        raise SystemExit(1)
