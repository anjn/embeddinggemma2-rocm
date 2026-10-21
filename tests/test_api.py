import base64, io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from embeddinggemma2_rocm.server import create_app


class FakeEmbedder:
    def __init__(self, *args):
        self.closed = False
        self.bundle = Path("/nonexistent")

    def close(self):
        self.closed = True

    @property
    def capabilities(self):
        return {"profiles": []}

    def tokenize(self, texts, prompt="query"):
        if prompt not in ("query", "document", None):
            raise ValueError("Invalid prompt")
        return np.ones((len(texts), 3), np.int64), np.ones((len(texts), 3), np.int64)

    def encode_tokens(self, ids, mask):
        return self.vectors(len(ids))

    def encode_images(self, images, token_budget=70):
        if token_budget not in (70, 140, 280):
            raise ValueError("Invalid budget")
        return self.vectors(len(images))

    @staticmethod
    def vectors(n):
        x = np.zeros((n, 768), np.float32)
        x[:, 0] = 1
        return x


@pytest.fixture
def client():
    with TestClient(create_app("unused", embedder_factory=FakeEmbedder)) as c:
        yield c


def test_text(client):
    r = client.post(
        "/v1/embeddings",
        json={"model": "embeddinggemma-2", "input": ["hello", "日本語"]},
    )
    assert r.status_code == 200
    d = r.json()
    assert (
        len(d["data"]) == 2
        and len(d["data"][0]["embedding"]) == 768
        and d["usage"]["total_tokens"] == 6
    )


def test_base64(client):
    d = client.post(
        "/v1/embeddings", json={"input": "hello", "encoding_format": "base64"}
    ).json()
    assert len(base64.b64decode(d["data"][0]["embedding"])) == 768 * 4


def test_image(client):
    f = io.BytesIO()
    Image.new("RGB", (16, 9)).save(f, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(f.getvalue()).decode()
    r = client.post(
        "/v1/embeddings",
        json={
            "input": [{"content": [{"type": "image_url", "image_url": {"url": url}}]}]
        },
    )
    assert r.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"input": []},
        {"input": ["x"] * 129},
        {"input": ["x", {}]},
        {"input": "x", "dimensions": 256},
        {"input": "x", "model": "wrong"},
        {
            "input": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        }
                    ]
                }
            ]
        },
    ],
)
def test_invalid(client, payload):
    assert client.post("/v1/embeddings", json=payload).status_code == 400


def test_health(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/v1/models").json()["data"][0]["id"] == "embeddinggemma-2"


def test_request_limit():
    with TestClient(
        create_app("unused", embedder_factory=FakeEmbedder, max_body_bytes=10)
    ) as c:
        assert (
            c.post(
                "/v1/embeddings", json={"input": "larger than ten bytes"}
            ).status_code
            == 413
        )


def test_startup_failure_closes_native_context():
    class FailingEmbedder(FakeEmbedder):
        @property
        def capabilities(self):
            return {"profiles": [{"budget": 70, "aspect": [16, 9]}]}

        def encode_images(self, *args, **kwargs):
            raise RuntimeError("Warmup failed")

    model = FailingEmbedder()
    with pytest.raises(RuntimeError, match="Warmup failed"):
        with TestClient(create_app("unused", embedder_factory=lambda *args: model)):
            pass
    assert model.closed


def test_decoded_image_limit():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    with TestClient(
        create_app("unused", max_body_bytes=1024, embedder_factory=FakeEmbedder)
    ) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "input": [
                    {"content": [{"type": "image_url", "image_url": {"url": url}}]}
                ]
            },
        )
    assert response.status_code == 400


def test_shared_bundle_prepared_before_ready():
    class SharedFake(FakeEmbedder):
        prepared = 0

        @property
        def capabilities(self):
            return {"schema": 2, "profiles": []}

        def prepare(self):
            type(self).prepared += 1

    with TestClient(create_app("/unused", embedder_factory=SharedFake)) as client:
        assert SharedFake.prepared == 1
        assert client.get("/ready").status_code == 200
