"""Direct libllama/libmtmd adapter; no HTTP in benchmark timings."""

import ctypes as C
import numpy as np


class LlamaBackend:
    def __init__(
        self, library, model, mmproj, budget, batch, flash=True, vision_batch=None
    ):
        if not all((library, model, mmproj)):
            raise ValueError("llama needs --llama-library, --model and --mmproj")
        self.lib = C.CDLL(str(library))
        self.lib.eg2_llama_create.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_char_p,
            C.c_size_t,
        ]
        self.lib.eg2_llama_create.restype = C.c_void_p
        self.lib.eg2_llama_destroy.argtypes = [C.c_void_p]
        self.lib.eg2_llama_destroy.restype = None
        self.lib.eg2_llama_encode.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_void_p,
            C.c_char_p,
            C.c_size_t,
        ]
        self.lib.eg2_llama_encode.restype = C.c_int
        self.lib.eg2_llama_info.argtypes = [C.c_void_p, C.c_void_p]
        self.lib.eg2_llama_info.restype = C.c_int
        self.lib.eg2_llama_system.restype = C.c_char_p
        self.lib.eg2_llama_device.restype = C.c_char_p
        self.metadata = {
            "system": self.lib.eg2_llama_system().decode(),
            "flash_attention": flash,
            "vision_batch_limit": vision_batch or (4 if budget == 280 else 16),
            "source_revision": "f12aaa344c971556ec35be054d7aac1d901fd510",
            "precision": "BF16 GGUF weights; default FP16 K/V cache",
            "resize_min_tokens": {70: 57, 140: 111, 280: 249}[budget],
            "resize_max_tokens": budget,
        }
        self.expected_tokens = {70: 66, 140: 120, 280: 264}[budget]
        error = C.create_string_buffer(4096)
        self.ctx = self.lib.eg2_llama_create(
            str(model).encode(),
            str(mmproj).encode(),
            budget,
            batch,
            int(flash),
            self.metadata["vision_batch_limit"],
            error,
            len(error),
        )
        if not self.ctx:
            raise RuntimeError(error.value.decode())
        self.metadata["gpu_backend"] = self.lib.eg2_llama_device().decode()
        self.metadata["system"] = self.lib.eg2_llama_system().decode()

    def encode(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        n, h, w, _ = rgb.shape
        if (h, w) != (288, 512):
            raise NotImplementedError(
                "The comparison adapter requires prescaled 512x288 RGB"
            )
        out = np.empty((n, 768), np.float32)
        error = C.create_string_buffer(4096)
        if self.lib.eg2_llama_encode(
            self.ctx, rgb.ctypes.data, n, h, w, out.ctypes.data, error, len(error)
        ):
            raise RuntimeError(error.value.decode())
        info = np.empty(4, np.int32)
        self.lib.eg2_llama_info(self.ctx, info.ctypes.data)
        self.metadata.update(
            actual_image_tokens=int(info[0]),
            actual_sequence_length=int(info[1]),
            vision_calls=int(info[2]),
            backbone_batch=n,
        )
        if int(info[0]) != self.expected_tokens:
            raise NotImplementedError(
                f"Requested grid needs {self.expected_tokens} image tokens; backend produced {int(info[0])}"
            )
        return out

    def close(self):
        if self.ctx:
            self.lib.eg2_llama_destroy(self.ctx)
            self.ctx = None
