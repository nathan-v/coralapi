import io

from PIL import Image


def png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    return {"file": ("img.png", buf.getvalue(), "image/png")}


def test_metrics_disabled_by_default(client):
    # Default settings: no /metrics route.
    assert client.get("/metrics").status_code == 404


def test_metrics_enabled_exposes_endpoint(make_client):
    c = make_client(metrics_enabled=True)
    resp = c.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    # Gauges are always present once the endpoint renders.
    assert "coralapi_queue_depth" in body
    assert "coralapi_queue_pending" in body
    assert "coralapi_models_loaded" in body


def test_inference_is_counted(make_client, fake_model):
    c = make_client(metrics_enabled=True)
    r = c.post("/v1/vision/classify", params={"model": "m1"}, files=png())
    assert r.status_code == 200
    body = c.get("/metrics").text
    # One successful classify against model "m1" recorded.
    assert 'coralapi_inferences_total{model="m1",operation="classify",status="ok"}' in body
    assert "coralapi_inference_duration_seconds" in body


def test_failed_inference_is_counted_with_status(make_client):
    # Unknown model -> 404, recorded with the mapped status label.
    c = make_client(metrics_enabled=True)
    r = c.post("/v1/vision/detect", params={"model": "nope"}, files=png())
    assert r.status_code == 404
    body = c.get("/metrics").text
    assert 'operation="detect"' in body and 'status="404"' in body


def test_enabling_without_dependency_errors(monkeypatch, tmp_path):
    import pytest

    import coralapi.metrics as metrics_mod
    from coralapi.config import Settings
    from coralapi.main import create_app

    # create_app validates the dependency before the lifespan TPU probe runs.
    monkeypatch.setattr(metrics_mod, "AVAILABLE", False)
    with pytest.raises(RuntimeError, match="prometheus-client is not"):
        create_app(Settings(model_dir=tmp_path, metrics_enabled=True))
