import base64
import io

import pytest
from PIL import Image


def png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 128, 255)).save(buf, format="PNG")
    return {"file": ("img.png", buf.getvalue(), "image/png")}


# -- segmentation -----------------------------------------------------------------


def test_segment_returns_mask_and_classes(client, fake_model):
    resp = client.post("/v1/vision/segment", params={"model": "deeplab"}, files=png())
    assert resp.status_code == 200
    body = resp.json()
    assert (body["width"], body["height"]) == (3, 2)
    # classes present sorted by pixel count desc: class 1 has 3 px, then 0 (2), 2 (1)
    assert [c["index"] for c in body["classes"]] == [1, 0, 2]
    assert body["classes"][0]["pixels"] == 3
    # mask_png decodes to an 8-bit L image of the mask size
    img = Image.open(io.BytesIO(base64.b64decode(body["mask_png"])))
    assert img.mode == "L" and img.size == (3, 2)
    assert img.getpixel((2, 1)) == 1  # bottom-right pixel is class 1


def test_segment_unknown_model_404(client):
    assert client.post("/v1/vision/segment", params={"model": "x"}, files=png()).status_code == 404


# -- pose -------------------------------------------------------------------------


def test_pose_returns_named_keypoints(client, fake_model):
    resp = client.post("/v1/vision/pose", params={"model": "movenet"}, files=png())
    assert resp.status_code == 200
    kps = resp.json()["keypoints"]
    assert len(kps) == 17
    assert kps[0]["name"] == "nose"
    assert kps[5]["name"] == "left_shoulder"
    # fake returns y=0.1*i, x=0.2, score=0.9 (float32, so compare approximately)
    assert kps[3]["x"] == pytest.approx(0.2, abs=1e-5)
    assert kps[3]["score"] == pytest.approx(0.9, abs=1e-5)


# -- embeddings -------------------------------------------------------------------


def test_embed_returns_vector(client, fake_model):
    resp = client.post("/v1/vision/embed", params={"model": "emb"}, files=png())
    assert resp.status_code == 200
    body = resp.json()
    assert body["dimension"] == 2
    assert body["normalized"] is False
    assert body["embedding"] == [3.0, 4.0]


def test_embed_normalized(client, fake_model):
    resp = client.post(
        "/v1/vision/embed", params={"model": "emb", "normalize": "true"}, files=png()
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["normalized"] is True
    assert body["embedding"] == pytest.approx([0.6, 0.8], abs=1e-5)  # [3,4]/5


def test_new_endpoints_documented(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/v1/vision/segment", "/v1/vision/pose", "/v1/vision/embed"):
        assert p in paths
