import httpx
import pytest

from coralapi.runtime.registry import (
    ModelDownloadError,
    ModelNotFound,
    ModelRegistry,
    zoo_label_file,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu", "coco_labels.txt"),
        ("efficientdet_lite0_320_ptq_edgetpu", "coco_labels.txt"),
        ("yolo11n_full_integer_quant_edgetpu", "coco_labels.txt"),
        ("mobilenet_v2_1.0_224_quant_edgetpu", "imagenet_labels.txt"),
        ("inception_v3_299_quant_edgetpu", "imagenet_labels.txt"),
        ("mobilenet_v2_1.0_224_inat_bird_quant_edgetpu", "inat_bird_labels.txt"),
        ("deeplabv3_mnv2_pascal_quant_edgetpu", "pascal_voc_segmentation_labels.txt"),
        ("ssd_mobilenet_v2_face_quant_postprocess_edgetpu", None),
        ("movenet_single_pose_lightning_ptq_edgetpu", None),
        ("some_unrecognized_model", None),
    ],
)
def test_zoo_label_file_heuristic(name, expected):
    assert zoo_label_file(name) == expected


def test_ensure_local_autofetches_shared_label_file(registry, monkeypatch):
    fetched = []

    def fake_fetch(self, url, dest):
        fetched.append(url)
        dest.write_bytes(b"x")
        return True

    monkeypatch.setattr(ModelRegistry, "_fetch", fake_fetch)
    name = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu"
    registry._ensure_local(name)
    assert "https://example.test/models/coco_labels.txt" in fetched
    assert registry._labels_path(name).name == "coco_labels.txt"


def test_user_labels_take_precedence_over_zoo_guess(registry):
    registry.model_dir.mkdir(exist_ok=True)
    (registry.model_dir / "m_coco.labels.txt").write_text("0 custom\n")
    (registry.model_dir / "coco_labels.txt").write_text("0 person\n")
    assert registry._labels_path("m_coco").name == "m_coco.labels.txt"


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path, model_source="https://example.test/models", model_download=True)


def test_http_source_rejected_by_default(tmp_path):
    with pytest.raises(ValueError, match="must be https"):
        ModelRegistry(tmp_path, model_source="http://example.test/models", model_download=True)


def test_http_source_allowed_when_opted_in(tmp_path):
    reg = ModelRegistry(
        tmp_path,
        model_source="http://example.test/models",
        model_download=True,
        allow_insecure_source=True,
    )
    assert reg.model_download is True


def test_fetch_refuses_redirect(registry, monkeypatch):
    class FakeResp:
        status_code = 302
        is_redirect = True
        headers = {"location": "http://169.254.169.254/latest/meta-data/"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: FakeResp())
    with pytest.raises(ModelDownloadError, match="Refusing redirect"):
        registry._fetch("https://example.test/models/x.tflite", registry.model_dir / "x")
    assert not (registry.model_dir / "x").exists()


def test_fetch_disables_redirect_following(registry, monkeypatch):
    seen = {}

    def fake_stream(method, url, **kwargs):
        seen.update(kwargs)
        raise httpx.ConnectError("stop here")

    monkeypatch.setattr(httpx, "stream", fake_stream)
    with pytest.raises(ModelDownloadError):
        registry._fetch("https://example.test/models/x.tflite", registry.model_dir / "x")
    assert seen["follow_redirects"] is False


def test_download_bounded_by_cache_limit(tmp_path, monkeypatch):
    reg = ModelRegistry(
        tmp_path,
        model_source="https://example.test/models",
        model_download=True,
        max_downloaded_models=2,
    )
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "a.tflite").write_bytes(b"x")
    (tmp_path / "b.tflite").write_bytes(b"x")
    monkeypatch.setattr(ModelRegistry, "_fetch", lambda self, url, dest: pytest.fail("no download"))
    with pytest.raises(ModelDownloadError, match="cache is full"):
        reg._ensure_local("c")


def test_download_fetches_missing_model(registry, monkeypatch):
    fetched = []

    def fake_fetch(self, url, dest):
        fetched.append(url)
        if url.endswith("foo.tflite"):
            dest.write_bytes(b"model")
            return True
        return False  # no labels sidecar upstream

    monkeypatch.setattr(ModelRegistry, "_fetch", fake_fetch)
    path = registry._ensure_local("foo")
    assert path.read_bytes() == b"model"
    assert fetched[0] == "https://example.test/models/foo.tflite"


def test_download_404_maps_to_model_not_found(registry, monkeypatch):
    monkeypatch.setattr(ModelRegistry, "_fetch", lambda self, url, dest: False)
    with pytest.raises(ModelNotFound):
        registry._ensure_local("missing")


def test_download_disabled_means_local_only(tmp_path):
    registry = ModelRegistry(tmp_path, model_source=None, model_download=True)
    with pytest.raises(ModelNotFound):
        registry._ensure_local("anything")


def test_local_model_wins_over_download(registry, tmp_path, monkeypatch):
    (tmp_path / "local.tflite").write_bytes(b"local")
    monkeypatch.setattr(
        ModelRegistry,
        "_fetch",
        lambda self, url, dest: pytest.fail("should not download"),
    )
    assert registry._ensure_local("local").read_bytes() == b"local"


def test_invalid_model_names_rejected(registry):
    for name in ("a/b", "../../etc/passwd", "", "a b"):
        with pytest.raises(ModelNotFound):
            registry.get(name)
