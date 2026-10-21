"""Optional OpenAI-style HTTP adapter; all inference goes through libeg2."""

from __future__ import annotations
import asyncio, base64, binascii, io, json, os
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from .runtime import Embedder, UnsupportedProfile


def create_app(
    bundle,
    device=0,
    library=None,
    *,
    max_pending=32,
    max_body_bytes=256 * 1024 * 1024,
    embedder_factory=Embedder,
):
    state = {"model": None, "pending": 0, "ready": False}
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        state["model"] = await asyncio.to_thread(
            embedder_factory, bundle, device, library
        )
        try:
            if state["model"].capabilities.get("schema") == 2:
                await asyncio.to_thread(state["model"].prepare)
            if state["model"].capabilities.get("text_profiles"):
                await asyncio.to_thread(
                    state["model"].encode_texts, ["warmup"], prompt="query"
                )
            profiles = state["model"].capabilities.get("profiles", [])
            if profiles:
                profile = profiles[0]
                aspect = profile.get("aspect") or [16, 9]
                sample = np.zeros((1, aspect[1] * 16, aspect[0] * 16, 3), np.uint8)
                await asyncio.to_thread(
                    state["model"].encode_images, sample, token_budget=profile["budget"]
                )
            state["ready"] = True
            yield
        finally:
            state["ready"] = False
            await asyncio.to_thread(state["model"].close)

    app = FastAPI(title="EmbeddingGemma 2 ROCm", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if not state["ready"]:
            raise HTTPException(503, "Model is not loaded")
        capabilities = await asyncio.to_thread(lambda: state["model"].capabilities)
        return {"status": "ready", "capabilities": capabilities}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": "embeddinggemma-2", "object": "model", "owned_by": "local"}
            ],
        }

    def infer(data):
        if data.get("model", "embeddinggemma-2") not in (
            "embeddinggemma-2",
            "embeddinggemma2",
        ):
            raise ValueError("Unknown model")
        if data.get("encoding_format", "float") not in ("float", "base64"):
            raise ValueError("encoding_format must be float or base64")
        if data.get("dimensions", 768) != 768:
            raise ValueError("Only 768-dimensional output is supported")
        inputs = data.get("input")
        inputs = [inputs] if isinstance(inputs, (str, dict)) else inputs
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= 128:
            raise ValueError("input must contain 1..128 items")
        model = state["model"]
        if all(isinstance(x, str) for x in inputs):
            ids, mask = model.tokenize(inputs, prompt=data.get("input_type", "query"))
            out = model.encode_tokens(ids, mask)
            tokens = int(mask.sum())
        elif all(isinstance(x, dict) for x in inputs):
            images = []
            decoded_bytes = 0
            for item in inputs:
                content = item.get("content", [])
                if (
                    not isinstance(content, list)
                    or len(content) != 1
                    or not isinstance(content[0], dict)
                    or content[0].get("type") != "image_url"
                ):
                    raise ValueError(
                        "Each image item requires exactly one image_url content entry"
                    )
                location = content[0].get("image_url", {})
                if not isinstance(location, dict):
                    raise ValueError("image_url must be an object")
                url = location.get("url", "")
                if not isinstance(url, str):
                    raise ValueError("Image URL must be a string")
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("Images must be base64 image data URLs")
                raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    decoded_bytes += image.width * image.height * 3
                    if decoded_bytes > max_body_bytes or max(image.size) > 32768:
                        raise ValueError("Decoded images exceed the request limit")
                    images.append(np.asarray(image.convert("RGB")))
            if len({im.shape for im in images}) != 1:
                raise ValueError("Images in one request must have equal dimensions")
            budget = data.get("token_budget", 70)
            out = model.encode_images(np.stack(images), token_budget=budget)
            tokens = 0
            tokens = int(
                model.capabilities.get("last_execution", {}).get("sequence", 0)
            ) * len(images)
        else:
            raise ValueError("Use a homogeneous batch of text strings or image objects")
        encoded = data.get("encoding_format", "float") == "base64"
        return {
            "object": "list",
            "model": "embeddinggemma-2",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": base64.b64encode(
                        np.asarray(row, dtype="<f4").tobytes()
                    ).decode()
                    if encoded
                    else row.tolist(),
                }
                for i, row in enumerate(out)
            ],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        if not state["ready"]:
            raise HTTPException(503, "Model is not loaded")
        if state["pending"] >= max_pending:
            raise HTTPException(429, "GPU request queue is full")
        state["pending"] += 1
        try:
            parts = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > max_body_bytes:
                    raise HTTPException(413, "Request body is too large")
                parts.append(chunk)
            data = json.loads(b"".join(parts))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            async with lock:
                return await asyncio.to_thread(infer, data)
        except UnsupportedProfile as e:
            raise HTTPException(503, str(e)) from e
        except (
            ValueError,
            TypeError,
            KeyError,
            binascii.Error,
            Image.UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as e:
            raise HTTPException(400, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(500, str(e)) from e
        finally:
            state["pending"] -= 1

    return app
