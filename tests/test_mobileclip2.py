"""Set MC2_TEST_BUNDLE and MC2_LIBRARY_PATH for native hardware tests."""

import ctypes as C
import json
import os
from pathlib import Path

import numpy as np
import pytest
from embeddinggemma2_rocm import MobileCLIP2Embedder, UnsupportedProfile


def test_backend_validation():
    with pytest.raises(ValueError, match="backend"):
        MobileCLIP2Embedder("unused", backend="cpu")


def test_text_validation():
    for value in ([], [1], ["a\0b"]):
        with pytest.raises(ValueError):
            MobileCLIP2Embedder._texts(value)
    assert len(MobileCLIP2Embedder._texts("日本語")) == 1


@pytest.fixture(scope="module")
def devices():
    bundle = os.environ.get("MC2_TEST_BUNDLE")
    if not bundle:
        pytest.skip("MC2_TEST_BUNDLE not set")
    # ROCr 7.2 probes /dev/accel: acquire it with XRT before HIP initializes.
    with (
        MobileCLIP2Embedder(bundle, "npu") as npu,
        MobileCLIP2Embedder(bundle, "gpu") as gpu,
    ):
        npu.prepare()
        gpu.prepare()
        yield {"npu": npu, "gpu": gpu}


@pytest.fixture(scope="module", params=["gpu", "npu"])
def model(request, devices):
    return devices[request.param]


def test_capabilities(model):
    info = model.capabilities
    assert info["dimension"] == 512
    assert info["image_batches"] == ([1, 4] if model.backend == "gpu" else [1])
    assert info["text_backend"] == "cpu-fp32"


def test_text_reference(model):
    cases = json.loads((model.bundle / "text-cases.json").read_text())
    ref = np.load(model.bundle / "text-reference.npy")
    y = model.encode_texts(cases)
    assert np.min(np.sum(y * ref, axis=1)) > 0.9999
    np.testing.assert_allclose(y, model.encode_tokens(model.tokenize(cases)), atol=1e-6)
    assert (
        model.tokenize(["a &amp;amp; b"]).tolist() == model.tokenize(["a & b"]).tolist()
    )
    assert model.tokenize(["person " * 100])[0, -1] == 49407


def test_images_and_errors(model):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, (192, 108, 3), dtype=np.uint8)
    y = model.encode_images([x])
    assert y.shape == (1, 512)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), 1, atol=1e-6)
    # Verify varied dimensions and a noncontiguous RGB view are accepted.
    if model.backend == "gpu":
        z = model.encode_images([x, x[::-1], x[:100], x[:64, :32]])
        assert z.shape == (4, 512)
        assert np.dot(y[0], z[0]) > 0.999
    for n in (0, 2, 3, 5, 32):
        with pytest.raises(UnsupportedProfile):
            model.encode_images([x] * n)
    if model.backend == "npu":
        with pytest.raises(UnsupportedProfile):
            model.encode_images([x] * 4)
    for bad in (x.astype(np.float32), x[:, :, 0], x[:0]):
        with pytest.raises(ValueError):
            model.encode_images([bad])
    with pytest.raises(ValueError):
        model.encode_pixels(np.full((1, 3, 256, 256), np.nan, np.float32))
    with pytest.raises(ValueError):
        model.encode_tokens(np.full((1, 77), 49408, np.int64))


def test_preprocessing_parity(model):
    from PIL import Image

    rng = np.random.default_rng(2)
    for h, w in [(192, 108), (513, 293), (32, 19), (256, 256), (1, 1)]:
        x = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        side = max(h, w)
        canvas = Image.new("RGB", (side, side))
        canvas.paste(Image.fromarray(x), ((side - w) // 2, (side - h) // 2))
        expected = (
            np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR))
            .transpose(2, 0, 1)
            .astype(np.float32)[None]
            / 255
        )
        direct = model.encode_images([x])
        pixels = model.encode_pixels(expected)
        np.testing.assert_allclose(direct, pixels, atol=1e-6, rtol=1e-5)


def test_c_errors(model):
    error = C.create_string_buffer(4096)
    y = np.empty((1, 512), np.float32)
    # No input access should happen when capacity is insufficient.
    status = model._lib.mc2_encode_images(
        model._ctx, None, 1, y.ctypes.data, 511, error, len(error)
    )
    assert status == 1
    status = model._lib.mc2_prepare(None, error, len(error))
    assert status == 1


def test_close():
    bundle = os.environ.get("MC2_TEST_BUNDLE")
    if not bundle:
        pytest.skip("MC2_TEST_BUNDLE not set")
    m = MobileCLIP2Embedder(bundle, "gpu")
    m.close()
    m.close()
    with pytest.raises(RuntimeError, match="closed"):
        m.encode_texts("hello")


def test_both_backends_same_process(devices):
    cwd = Path.cwd()
    gpu, npu = devices["gpu"], devices["npu"]
    x = np.full((1, 3, 256, 256), 0.5, np.float32)
    a = gpu.encode_pixels(x)
    b = npu.encode_pixels(x)
    c = gpu.encode_pixels(x)
    np.testing.assert_allclose(a, c, atol=1e-6)
    assert np.isfinite(b).all()
    assert Path.cwd() == cwd


def test_tokenizer_reference(model):
    cases = json.loads((model.bundle / "tokenizer-cases.json").read_text())
    actual = model.tokenize([c["text"] for c in cases])
    np.testing.assert_array_equal(actual, np.array([c["ids"] for c in cases], np.int64))


def test_native_stride(model):
    from embeddinggemma2_rocm.mobileclip2 import _Image

    rng = np.random.default_rng(3)
    storage = rng.integers(0, 256, (27, 64), dtype=np.uint8)
    expected = model.encode_images(storage[:, :57].reshape(27, 19, 3))
    descriptor = _Image(storage.ctypes.data, 19, 27, 64)
    actual = np.empty((1, 512), np.float32)
    error = C.create_string_buffer(4096)
    status = model._lib.mc2_encode_images(
        model._ctx, C.byref(descriptor), 1, actual.ctypes.data, 512, error, len(error)
    )
    assert status == 0, error.value
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_bundle_mismatch(devices, tmp_path):
    root = devices["gpu"].bundle
    for path in root.iterdir():
        if path.name != "manifest.json":
            (tmp_path / path.name).symlink_to(
                path.resolve(), target_is_directory=path.is_dir()
            )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["dimension"] = 768
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible"):
        MobileCLIP2Embedder(tmp_path, "gpu")
    manifest["dimension"] = 512
    manifest["image_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        MobileCLIP2Embedder(tmp_path, "gpu")
