import io

from PIL import Image


def png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 128, 255)).save(buf, format="PNG")
    return {"file": ("img.png", buf.getvalue(), "image/png")}


def test_raw_returns_output_tensors(client, fake_model):
    resp = client.post("/v1/raw/infer", params={"model": "anymodel"}, files=png())
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "anymodel"
    assert body["input_shape"] == [1, 224, 224, 3]
    out = body["outputs"][0]
    assert out["name"] == "output0"
    assert out["shape"] == [1, 3]
    assert out["dtype"] == "float32"
    assert out["quantization"] == {"scale": 0.00392, "zero_point": 0}
    assert out["data"] == [0.1, 0.2, 0.7]


def test_raw_dequantize_false_is_passed_through(client, fake_model):
    resp = client.post(
        "/v1/raw/infer",
        params={"model": "m", "dequantize": "false"},
        files=png(),
    )
    assert resp.status_code == 200
    assert resp.json()["outputs"][0]["dtype"] == "uint8"


def test_raw_unknown_model_404(client):
    resp = client.post("/v1/raw/infer", params={"model": "nope"}, files=png())
    assert resp.status_code == 404


def test_raw_rejects_non_image(client, fake_model):
    resp = client.post(
        "/v1/raw/infer",
        params={"model": "m"},
        files={"file": ("f.bin", b"not an image", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_raw_documented_in_openapi(client):
    spec = client.get("/openapi.json").json()
    assert "/v1/raw/infer" in spec["paths"]
