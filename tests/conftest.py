import numpy as np
import pytest
from fastapi.testclient import TestClient

import coralapi.runtime.device as device_mod
import coralapi.runtime.tflite as tflite_mod
from coralapi.config import Settings
from coralapi.main import create_app
from coralapi.runtime.engine import LoadedModel
from coralapi.runtime.registry import ModelRegistry
from coralapi.schemas import Classification, Detection, DeviceInfo

FAKE_DEVICE = DeviceInfo(type="usb", path="/sys/bus/usb/devices/fake")


class FakeInterpreter:
    """Stand-in for a tflite Interpreter to exercise LoadedModel preprocessing
    without a TPU: captures the tensor handed to set_tensor and returns canned
    outputs from get_tensor."""

    def __init__(self, input_detail, outputs=None):
        self._input_detail = input_detail
        self._outputs = outputs or []  # list of (detail_dict, ndarray)
        self.set_tensors = {}  # index -> the array that was set

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [self._input_detail]

    def get_output_details(self):
        return [detail for detail, _ in self._outputs]

    def set_tensor(self, index, data):
        self.set_tensors[index] = data

    def invoke(self):
        pass

    def get_tensor(self, index):
        for detail, arr in self._outputs:
            if detail["index"] == index:
                return arr
        raise KeyError(index)


@pytest.fixture
def loaded_model():
    """Factory building a real LoadedModel wired to a FakeInterpreter (no TPU).

    Bypasses __init__ (which would create a delegate) and sets only the
    attributes the methods use, so _preprocess / input_size / detect run the
    real code.
    """

    def _make(shape, dtype, quantization=(0.0, 0), outputs=None, labels=None):
        input_detail = {
            "shape": np.array(shape),
            "dtype": dtype,
            "quantization": quantization,
            "index": 0,
        }
        model = LoadedModel.__new__(LoadedModel)
        model.name = "fake"
        model.path = None
        model.labels = labels or {}
        model.interpreter = FakeInterpreter(input_detail, outputs)
        model.input_detail = input_detail
        model.output_details = model.interpreter.get_output_details()
        return model

    return _make


@pytest.fixture
def fake_tpu(monkeypatch):
    """Pretend an Edge TPU delegate loads and one USB device is present, so the
    app can start (and pass readiness) with no real hardware."""
    monkeypatch.setattr(tflite_mod, "load_edgetpu_delegate", lambda device, lib=None: object())
    monkeypatch.setattr(device_mod, "discover_all", lambda: [FAKE_DEVICE])


@pytest.fixture
def make_client(fake_tpu, tmp_path):
    """Factory for TestClients with per-test settings overrides."""
    clients = []

    def _make(**overrides) -> TestClient:
        defaults = dict(model_dir=tmp_path / "models", model_download=False)
        client = TestClient(create_app(Settings(**{**defaults, **overrides})))
        client.__enter__()  # run lifespan (the startup TPU check)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


class FakeModel:
    """Stands in for a TPU-loaded model in API-level tests."""

    input_detail = {"shape": [1, 224, 224, 3]}
    labels = {0: "background", 1: "person", 2: "car"}

    def classify(self, image, top_k=5):
        return [Classification(index=1, label="cat", score=0.9)][:top_k]

    def detect(self, image, threshold=0.4):
        return [Detection(box=[0.1, 0.1, 0.9, 0.9], index=1, label="cat", score=0.8)]

    def raw(self, image, dequantize=True):
        return [
            {
                "name": "output0",
                "shape": [1, 3],
                "dtype": "float32" if dequantize else "uint8",
                "quantization": {"scale": 0.00392, "zero_point": 0},
                "data": [0.1, 0.2, 0.7],
            }
        ]

    def segment(self, image):
        import numpy as np

        # 2x3 class map: classes 0, 1, 2 present.
        return np.array([[0, 0, 1], [2, 1, 1]], dtype=np.int64)

    def pose(self, image):
        import numpy as np

        # 17 keypoints of (y, x, score).
        return np.array([[0.1 * i, 0.2, 0.9] for i in range(17)], dtype=np.float32)

    def embed(self, image, normalize=False):
        import numpy as np

        v = np.array([3.0, 4.0], dtype=np.float32)
        return v / 5.0 if normalize else v


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(ModelRegistry, "get", lambda self, name: model)
    return model
