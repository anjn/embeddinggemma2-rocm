from __future__ import annotations
import ctypes as C
import json
import os
from pathlib import Path
import threading
import numpy as np


class UnsupportedProfile(RuntimeError):
    """No prepared profile exists for the requested input shape."""


def _library(path=None):
    candidates = [
        path,
        os.environ.get("EG2_LIBRARY_PATH"),
        Path(__file__).with_name("libeg2.so"),
        "libeg2.so.0",
        "libeg2.so",
    ]
    failures = []
    for p in candidates:
        if not p:
            continue
        try:
            return C.CDLL(str(p))
        except OSError as e:
            failures.append(str(e))
    raise RuntimeError(
        "Cannot load libeg2. Set EG2_LIBRARY_PATH. " + "; ".join(failures)
    )


class Embedder:
    """A serialized native execution context. Use a separate context per GPU worker.

    Images are RGB uint8 arrays; image decoding and external prescaling are caller work.
    The first call for a prepared shape performs warmup and graph capture.
    """

    def __init__(self, bundle, device=0, library=None):
        self._lib = _library(library)
        self._ctx = C.c_void_p()
        self._lock = threading.RLock()
        self._lib.eg2_create.argtypes = [
            C.c_char_p,
            C.c_int,
            C.POINTER(C.c_void_p),
            C.c_char_p,
            C.c_size_t,
        ]
        self._lib.eg2_create.restype = C.c_int
        self._lib.eg2_destroy.argtypes = [C.c_void_p]
        self._lib.eg2_destroy.restype = None
        self._lib.eg2_encode_images.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.c_size_t,
            C.c_size_t,
            C.c_int,
            C.c_void_p,
            C.c_size_t,
            C.c_char_p,
            C.c_size_t,
        ]
        self._lib.eg2_encode_images.restype = C.c_int
        self._lib.eg2_encode_tokens.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_char_p,
            C.c_size_t,
        ]
        self._lib.eg2_encode_tokens.restype = C.c_int
        self._lib.eg2_info.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
        self._lib.eg2_info.restype = C.c_int
        self._lib.eg2_tokenize.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_char_p),
            C.c_size_t,
            C.c_char_p,
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_char_p,
            C.c_size_t,
        ]
        self._lib.eg2_tokenize.restype = C.c_int
        self.bundle = Path(bundle).resolve()
        error = C.create_string_buffer(4096)
        self._check(
            self._lib.eg2_create(
                os.fsencode(self.bundle), device, C.byref(self._ctx), error, len(error)
            ),
            error,
        )

    @staticmethod
    def _check(status, error):
        if status:
            message = error.value.decode("utf-8", errors="replace")
            raise (
                ValueError
                if status == 1
                else UnsupportedProfile
                if status == 3
                else RuntimeError
            )(message)

    def _open(self):
        if not self._ctx:
            raise RuntimeError("Embedder is closed")

    def close(self):
        with self._lock:
            if self._ctx:
                self._lib.eg2_destroy(self._ctx)
                self._ctx = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, "_ctx", None):
            self.close()

    @property
    def capabilities(self):
        with self._lock:
            self._open()
            buf = C.create_string_buffer(1024 * 1024)
            if self._lib.eg2_info(self._ctx, buf, len(buf)):
                raise RuntimeError("Capability response too large")
            return json.loads(buf.value)

    def encode_images(self, images, *, token_budget=70):
        if not isinstance(images, np.ndarray):
            from PIL import Image

            images = np.stack(
                [
                    np.asarray(im.convert("RGB") if isinstance(im, Image.Image) else im)
                    for im in images
                ]
            )
        if images.ndim == 3:
            images = images[None]
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("Expected uint8 RGB [N,H,W,3]")
        images = np.ascontiguousarray(images)
        n, h, w, _ = images.shape
        out = np.empty((n, 768), np.float32)
        error = C.create_string_buffer(4096)
        with self._lock:
            self._open()
            self._check(
                self._lib.eg2_encode_images(
                    self._ctx,
                    images.ctypes.data,
                    n,
                    h,
                    w,
                    token_budget,
                    out.ctypes.data,
                    out.size,
                    error,
                    len(error),
                ),
                error,
            )
        return out

    def encode_tokens(self, input_ids, attention_mask):
        ids = np.ascontiguousarray(input_ids, dtype=np.int64)
        mask = np.ascontiguousarray(attention_mask, dtype=np.int64)
        if ids.ndim != 2 or ids.shape != mask.shape:
            raise ValueError("Expected equal [batch,sequence] arrays")
        out = np.empty((len(ids), 768), np.float32)
        error = C.create_string_buffer(4096)
        with self._lock:
            self._open()
            self._check(
                self._lib.eg2_encode_tokens(
                    self._ctx,
                    ids.ctypes.data,
                    mask.ctypes.data,
                    *ids.shape,
                    out.ctypes.data,
                    out.size,
                    error,
                    len(error),
                ),
                error,
            )
        return out

    def tokenize(self, texts, *, prompt="query", max_length=8192):
        if isinstance(texts, str):
            texts = [texts]
        if not 1 <= len(texts) <= 128 or not 1 <= max_length <= 8192:
            raise ValueError("Text batch must be 1..128 and max_length 1..8192")
        prefixes = {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
            None: "",
        }
        if prompt not in prefixes:
            raise ValueError("prompt must be query, document or None")
        if any(not isinstance(t, str) or "\0" in t for t in texts):
            raise ValueError("Texts must be UTF-8 strings without NUL")
        values = (C.c_char_p * len(texts))(*(t.encode("utf-8") for t in texts))
        ids = np.zeros((len(texts), max_length), np.int64)
        mask = np.zeros_like(ids)
        lengths = np.empty(len(texts), np.uintp)
        error = C.create_string_buffer(4096)
        with self._lock:
            self._open()
            self._check(
                self._lib.eg2_tokenize(
                    self._ctx,
                    values,
                    len(texts),
                    prefixes[prompt].encode(),
                    ids.ctypes.data,
                    mask.ctypes.data,
                    max_length,
                    lengths.ctypes.data,
                    error,
                    len(error),
                ),
                error,
            )
        length = int(lengths.max())
        return np.ascontiguousarray(ids[:, :length]), np.ascontiguousarray(
            mask[:, :length]
        )

    def encode_texts(self, texts, *, prompt="query", max_length=8192):
        ids, mask = self.tokenize(texts, prompt=prompt, max_length=max_length)
        return self.encode_tokens(ids, mask)
