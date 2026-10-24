"""MobileCLIP2 C ABI wrapper. Inference does not import torch or onnxruntime."""

from __future__ import annotations

import ctypes as C
import json
import os
import threading
from pathlib import Path

import numpy as np

from .runtime import UnsupportedProfile


class _Image(C.Structure):
    _fields_ = [
        ("data", C.c_void_p),
        ("width", C.c_size_t),
        ("height", C.c_size_t),
        ("row_stride", C.c_size_t),
    ]


class MobileCLIP2Embedder:
    """Normalized 512D vectors; GPU image batches 1/4, NPU image batch 1.

    Images are RGB uint8 arrays, individually sized, at most 16384 per side.
    Text always runs on CPU. Calls serialize; close() releases native resources.
    """

    def __init__(self, bundle, backend="gpu", library=None):
        if backend not in ("gpu", "npu"):
            raise ValueError("backend must be gpu or npu")
        self._lock = threading.RLock()
        self._ctx = C.c_void_p()
        self.bundle = Path(bundle).resolve()
        self.backend = backend
        candidates = [
            library,
            os.environ.get("MC2_LIBRARY_PATH"),
            Path(__file__).with_name("libmobileclip2.so"),
            "libmobileclip2.so.0",
            "libmobileclip2.so",
        ]
        errors = []
        for candidate in filter(None, candidates):
            try:
                self._lib = C.CDLL(str(candidate))
                break
            except OSError as e:
                errors.append(str(e))
        else:
            raise RuntimeError(
                "Cannot load libmobileclip2. Set MC2_LIBRARY_PATH. " + "; ".join(errors)
            )
        ptr = C.c_void_p
        size = C.c_size_t
        err = [C.c_char_p, size]
        signatures = {
            "create": [C.c_char_p, C.c_char_p, C.POINTER(ptr), *err],
            "prepare": [ptr, *err],
            "encode_images": [ptr, C.POINTER(_Image), size, ptr, size, *err],
            "encode_pixels": [ptr, ptr, size, ptr, size, *err],
            "encode_tokens": [ptr, ptr, size, ptr, size, *err],
            "encode_texts": [ptr, C.POINTER(C.c_char_p), size, ptr, size, *err],
            "tokenize": [ptr, C.POINTER(C.c_char_p), size, ptr, size, *err],
            "info": [ptr, C.c_char_p, size],
        }
        for name, args in signatures.items():
            f = getattr(self._lib, "mc2_" + name)
            f.argtypes = args
            f.restype = C.c_int
        self._lib.mc2_destroy.argtypes = [ptr]
        self._lib.mc2_destroy.restype = None
        self._call(
            "create",
            os.fsencode(self.bundle),
            backend.encode(),
            C.byref(self._ctx),
            check_open=False,
        )

    def _call(self, name, *args, check_open=True):
        if check_open and not self._ctx:
            raise RuntimeError("MobileCLIP2Embedder is closed")
        e = C.create_string_buffer(4096)
        status = getattr(self._lib, "mc2_" + name)(*args, e, len(e))
        if status:
            kind = (
                ValueError
                if status == 1
                else UnsupportedProfile
                if status == 3
                else RuntimeError
            )
            raise kind(e.value.decode("utf-8", errors="replace"))

    def close(self):
        with self._lock:
            if self._ctx:
                self._lib.mc2_destroy(self._ctx)
                self._ctx = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, "_ctx", None):
            self.close()

    def prepare(self):
        with self._lock:
            self._call("prepare", self._ctx)

    @property
    def capabilities(self):
        with self._lock:
            if not self._ctx:
                raise RuntimeError("MobileCLIP2Embedder is closed")
            buf = C.create_string_buffer(16384)
            status = self._lib.mc2_info(self._ctx, buf, len(buf))
            if status:
                raise RuntimeError("Cannot read MobileCLIP capabilities")
            return json.loads(buf.value)

    def _batch(self, n):
        allowed = (1, 4) if self.backend == "gpu" else (1,)
        if n not in allowed:
            raise UnsupportedProfile(
                f"{self.backend} image batch must be one of {allowed}"
            )

    def encode_images(self, images):
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        images = list(images)
        self._batch(len(images))
        arrays = []
        for im in images:
            a = np.asarray(im)
            if (
                a.dtype != np.uint8
                or a.ndim != 3
                or a.shape[2] != 3
                or not all(a.shape[:2])
            ):
                raise ValueError("Images must be nonempty RGB uint8 [height,width,3]")
            if max(a.shape[:2]) > 16384:
                raise ValueError("Image dimensions exceed 16384")
            arrays.append(np.ascontiguousarray(a))
        descriptors = (_Image * len(arrays))(
            *[
                _Image(a.ctypes.data, a.shape[1], a.shape[0], a.strides[0])
                for a in arrays
            ]
        )
        y = np.empty((len(arrays), 512), np.float32)
        with self._lock:
            self._call(
                "encode_images",
                self._ctx,
                descriptors,
                len(arrays),
                y.ctypes.data,
                y.size,
            )
        return y

    def encode_pixels(self, pixels):
        x = np.asarray(pixels)
        if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (3, 256, 256):
            raise ValueError("Expected float32 [batch,3,256,256]")
        self._batch(len(x))
        x = np.ascontiguousarray(x)
        y = np.empty((len(x), 512), np.float32)
        with self._lock:
            self._call(
                "encode_pixels", self._ctx, x.ctypes.data, len(x), y.ctypes.data, y.size
            )
        return y

    @staticmethod
    def _texts(texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts or any(not isinstance(t, str) or "\0" in t for t in texts):
            raise ValueError("Expected nonempty sequence of strings without NUL")
        return (C.c_char_p * len(texts))(*[t.encode("utf-8") for t in texts])

    def tokenize(self, texts):
        x = self._texts(texts)
        y = np.empty((len(x), 77), np.int64)
        with self._lock:
            self._call("tokenize", self._ctx, x, len(x), y.ctypes.data, y.size)
        return y

    def encode_tokens(self, tokens):
        x = np.asarray(tokens)
        if x.dtype != np.int64 or x.ndim != 2 or x.shape[1] != 77 or not len(x):
            raise ValueError("Expected int64 [count,77]")
        x = np.ascontiguousarray(x)
        y = np.empty((len(x), 512), np.float32)
        with self._lock:
            self._call(
                "encode_tokens", self._ctx, x.ctypes.data, len(x), y.ctypes.data, y.size
            )
        return y

    def encode_texts(self, texts):
        x = self._texts(texts)
        y = np.empty((len(x), 512), np.float32)
        with self._lock:
            self._call("encode_texts", self._ctx, x, len(x), y.ctypes.data, y.size)
        return y
