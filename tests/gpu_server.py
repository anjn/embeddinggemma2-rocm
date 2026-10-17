"""Real-GPU HTTP smoke/parity check, separate from CPU protocol tests."""

import argparse, base64, io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from embeddinggemma2_rocm import Embedder
from embeddinggemma2_rocm.runtime import UnsupportedProfile
from embeddinggemma2_rocm.server import create_app

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
a = p.parse_args()
rgb = np.zeros((288, 512, 3), np.uint8)
rgb[:] = [127, 95, 63]
with Embedder(a.bundle) as model:
    try:
        model.encode_images(np.zeros((33, 9, 16, 3), np.uint8), token_budget=280)
    except UnsupportedProfile as error:
        assert "32 images" in str(error), str(error)
    else:
        raise AssertionError("Budget-280 request limit did not reject batch 33")
    expected_image = model.encode_images(rgb)[0]
    expected_text = model.encode_texts(["日本語の検索"])[0]
buffer = io.BytesIO()
Image.fromarray(rgb).save(buffer, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
with TestClient(create_app(a.bundle)) as client:
    assert client.get("/ready").status_code == 200
    rejected = client.post(
        "/v1/embeddings",
        json={
            "input": [{"content": [{"type": "image_url", "image_url": {"url": url}}]}] * 33,
            "token_budget": 280,
        },
    )
    assert rejected.status_code == 503, rejected.text
    assert "32 images" in rejected.text, rejected.text
    image = client.post(
        "/v1/embeddings",
        json={
            "input": [{"content": [{"type": "image_url", "image_url": {"url": url}}]}],
            "token_budget": 70,
        },
    )
    assert image.status_code == 200, image.text
    np.testing.assert_allclose(
        image.json()["data"][0]["embedding"], expected_image, atol=1e-6, rtol=1e-5
    )
    text = client.post("/v1/embeddings", json={"input": ["日本語の検索"]})
    assert text.status_code == 200, text.text
    np.testing.assert_allclose(
        text.json()["data"][0]["embedding"], expected_text, atol=1e-6, rtol=1e-5
    )
print("GPU HTTP readiness, image/text requests and direct-call parity passed")
