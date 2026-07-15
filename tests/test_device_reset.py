"""Readiness accuracy and the /v1/reset rebind endpoint."""

import coralapi.runtime.device as device_mod
import coralapi.runtime.tflite as tflite_mod
from coralapi.schemas import DeviceInfo


def test_readyz_ok_when_device_present_and_unchanged(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["devices"] == ["/sys/bus/usb/devices/fake"]


def test_readyz_503_when_device_unplugged(client, monkeypatch):
    monkeypatch.setattr(device_mod, "discover_all", lambda: [])
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert "No Coral" in resp.json()["detail"]


def test_readyz_503_when_device_reenumerated(client, monkeypatch):
    # Bound at startup to .../fake; a replug shows a different path -> stale delegate.
    monkeypatch.setattr(
        device_mod, "discover_all", lambda: [DeviceInfo(type="usb", path="/sys/.../moved")]
    )
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert "reset" in resp.json()["detail"].lower()


def test_reset_rebinds_and_reports_devices(client):
    resp = client.post("/v1/reset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reset"] is True
    assert body["models_cleared"] == 0  # no models were loaded in this test
    assert body["devices"] == ["/sys/bus/usb/devices/fake"]


def test_reset_updates_binding_so_readyz_recovers(client, monkeypatch):
    # Device replugged to a new path -> readyz is 503 until we rebind.
    monkeypatch.setattr(
        device_mod, "discover_all", lambda: [DeviceInfo(type="usb", path="/sys/.../new")]
    )
    assert client.get("/readyz").status_code == 503
    assert client.post("/v1/reset").json()["devices"] == ["/sys/.../new"]
    assert client.get("/readyz").status_code == 200  # now bound to the new path


def test_reset_503_when_tpu_gone_after_reset(client, monkeypatch):
    def boom(device, lib=None):
        raise tflite_mod.DelegateUnavailable("device gone")

    monkeypatch.setattr(tflite_mod, "load_edgetpu_delegate", boom)
    resp = client.post("/v1/reset")
    assert resp.status_code == 503
