import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import coralapi.runtime.tflite as tflite_mod
from coralapi.config import Settings
from coralapi.main import create_app


def png_bytes(size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def upload(name="img.png", data=None, mime="image/png"):
    return {"file": (name, data if data is not None else png_bytes(), mime)}


# -- startup ------------------------------------------------------------------


def test_startup_without_tpu_fails_with_clear_error(tmp_path, monkeypatch):
    def no_tpu(device, lib=None):
        raise tflite_mod.DelegateUnavailable("libedgetpu.so.1 not found")

    monkeypatch.setattr(tflite_mod, "load_edgetpu_delegate", no_tpu)
    app = create_app(Settings(model_dir=tmp_path))
    with pytest.raises(RuntimeError, match="Coral Edge TPU unavailable"):
        with TestClient(app):
            pass


# -- system endpoints --------------------------------------------------------


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readyz(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["models_available"] == 0


def test_status_reports_runtime(client):
    body = client.get("/v1/status").json()
    assert set(body) == {"interpreter", "device_preference", "devices"}
    assert body["device_preference"] == "auto"


def test_queue_status(client):
    body = client.get("/v1/queue").json()
    assert body == {"depth": 16, "pending": 0}


def test_openapi_documents_v1(client):
    spec = client.get("/openapi.json").json()
    v1_paths = [p for p in spec["paths"] if p.startswith("/v1/")]
    assert "/v1/vision/classify" in v1_paths
    assert "/v1/jobs/classify" in v1_paths
    assert client.get("/docs").status_code == 200


# -- models --------------------------------------------------------------------


def test_list_models_empty(client):
    assert client.get("/v1/models").json() == []


def test_list_models_finds_tflite(client, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "ssd_mobilenet_edgetpu.tflite").write_bytes(b"\x00")
    (model_dir / "ssd_mobilenet_edgetpu.labels.txt").write_text("0 person\n")
    (model_dir / "notes.txt").write_text("ignored")

    models = client.get("/v1/models").json()
    assert len(models) == 1
    assert models[0]["name"] == "ssd_mobilenet_edgetpu"
    assert models[0]["edgetpu_compiled"] is True
    assert models[0]["has_labels"] is True
    assert models[0]["loaded"] is False


# -- sync inference -------------------------------------------------------------


def test_classify_unknown_model_404(client):
    resp = client.post("/v1/vision/classify", params={"model": "nope"}, files=upload())
    assert resp.status_code == 404


def test_classify_returns_results(client, fake_model):
    resp = client.post("/v1/vision/classify", params={"model": "fake"}, files=upload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "fake"
    assert body["results"][0]["label"] == "cat"


def test_detect_returns_results(client, fake_model):
    resp = client.post("/v1/vision/detect", params={"model": "fake"}, files=upload())
    assert resp.status_code == 200
    assert resp.json()["results"][0]["box"] == [0.1, 0.1, 0.9, 0.9]


def test_classify_rejects_non_image(client, fake_model):
    resp = client.post(
        "/v1/vision/classify",
        params={"model": "fake"},
        files=upload("f.bin", b"not an image", "application/octet-stream"),
    )
    assert resp.status_code == 400


# -- upload limits ----------------------------------------------------------------


def test_upload_over_size_limit_413(make_client, fake_model):
    client = make_client(max_upload_bytes=10)
    resp = client.post("/v1/vision/classify", params={"model": "fake"}, files=upload())
    assert resp.status_code == 413


def test_image_over_resolution_limit_413(make_client, fake_model):
    client = make_client(max_image_dim=4)
    resp = client.post(
        "/v1/vision/classify",
        params={"model": "fake"},
        files=upload(data=png_bytes(size=(8, 8))),
    )
    assert resp.status_code == 413
    assert "longest edge" in resp.json()["detail"]


# -- async jobs ---------------------------------------------------------------------


def poll_until_finished(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {body}")


def test_async_classify_job(client, fake_model):
    resp = client.post("/v1/jobs/classify", params={"model": "fake"}, files=upload())
    assert resp.status_code == 202
    job = resp.json()
    assert job["status"] in ("queued", "running")

    done = poll_until_finished(client, job["id"])
    assert done["status"] == "succeeded"
    assert done["result"]["results"][0]["label"] == "cat"


def test_async_job_failure_is_reported(client):
    # Unknown model: the job is accepted, then fails during execution.
    resp = client.post("/v1/jobs/classify", params={"model": "nope"}, files=upload())
    assert resp.status_code == 202
    done = poll_until_finished(client, resp.json()["id"])
    assert done["status"] == "failed"
    assert "nope" in done["error"]


def test_unknown_job_404(client):
    assert client.get("/v1/jobs/deadbeef").status_code == 404


# -- error leakage ----------------------------------------------------------------

SECRET_URL = "https://internal-mirror.corp.local/secret/path/model.tflite"


def _raise_download_error(client, monkeypatch):
    from coralapi.runtime.registry import ModelDownloadError, ModelRegistry

    def boom(self, name):
        raise ModelDownloadError(f"Failed to download {SECRET_URL}: connection refused")

    monkeypatch.setattr(ModelRegistry, "get", boom)


def test_sync_download_error_does_not_leak_url(client, monkeypatch):
    _raise_download_error(client, monkeypatch)
    resp = client.post("/v1/vision/detect", params={"model": "x"}, files=upload())
    assert resp.status_code == 502
    assert SECRET_URL not in resp.text
    assert "internal-mirror" not in resp.text


def test_async_download_error_does_not_leak_url(client, monkeypatch):
    _raise_download_error(client, monkeypatch)
    resp = client.post("/v1/jobs/detect", params={"model": "x"}, files=upload())
    assert resp.status_code == 202
    done = poll_until_finished(client, resp.json()["id"])
    assert done["status"] == "failed"
    assert SECRET_URL not in (done["error"] or "")


# -- queue backpressure -----------------------------------------------------------


def test_queue_full_rejects_with_429(make_client, monkeypatch):
    from coralapi.runtime.registry import ModelRegistry
    from coralapi.schemas import Classification

    release = threading.Event()

    class SlowModel:
        def classify(self, image, top_k=5):
            release.wait(timeout=5)
            return [Classification(index=0, label="slow", score=1.0)]

    monkeypatch.setattr(ModelRegistry, "get", lambda self, name: SlowModel())
    client = make_client(queue_depth=0)

    # Occupy the only slot with an async job...
    accepted = client.post("/v1/jobs/classify", params={"model": "slow"}, files=upload())
    assert accepted.status_code == 202
    assert client.get("/v1/queue").json()["pending"] == 1

    # ...so the next request finds the queue full.
    rejected = client.post("/v1/vision/classify", params={"model": "slow"}, files=upload())
    assert rejected.status_code == 429

    release.set()
    done = poll_until_finished(client, accepted.json()["id"])
    assert done["status"] == "succeeded"
    assert client.get("/v1/queue").json()["pending"] == 0
