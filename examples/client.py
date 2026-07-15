"""Minimal Python client for coralapi, with a decoding helper per endpoint.

    from examples.client import CoralAPI
    api = CoralAPI("http://localhost:8000")
    api.classify("cat.jpg", "tf2_mobilenet_v2_1.0_224_ptq_edgetpu", top_k=3)

Or run a quick demo against a live server:

    python examples/client.py http://localhost:8000 cat.jpg

Requires httpx (always) and, for the decoding helpers, numpy + pillow — all of
which `uv sync` installs.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import httpx


class CoralAPI:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _infer(self, path: str, image: str, params: dict) -> dict:
        data = Path(image).read_bytes()
        resp = self._client.post(path, params=params, files={"file": (Path(image).name, data)})
        resp.raise_for_status()
        return resp.json()

    # -- task endpoints ---------------------------------------------------

    def classify(self, image: str, model: str, top_k: int = 5) -> dict:
        return self._infer("/v1/vision/classify", image, {"model": model, "top_k": top_k})

    def detect(self, image: str, model: str, threshold: float = 0.4) -> dict:
        return self._infer("/v1/vision/detect", image, {"model": model, "threshold": threshold})

    def segment(self, image: str, model: str) -> dict:
        return self._infer("/v1/vision/segment", image, {"model": model})

    def pose(self, image: str, model: str) -> dict:
        return self._infer("/v1/vision/pose", image, {"model": model})

    def embed(self, image: str, model: str, normalize: bool = False) -> list[float]:
        params = {"model": model, "normalize": str(normalize).lower()}
        return self._infer("/v1/vision/embed", image, params)["embedding"]

    def raw(self, image: str, model: str, dequantize: bool = True) -> dict:
        params = {"model": model, "dequantize": str(dequantize).lower()}
        return self._infer("/v1/raw/infer", image, params)

    # -- decoding helpers -------------------------------------------------

    def segment_mask(self, image: str, model: str):
        """Return the segmentation mask as an (H, W) numpy array of class indices."""
        import numpy as np
        from PIL import Image

        png = base64.b64decode(self.segment(image, model)["mask_png"])
        return np.array(Image.open(io.BytesIO(png)))

    # -- system -----------------------------------------------------------

    def status(self) -> dict:
        return self._client.get("/v1/status").json()


def _demo(base_url: str, image: str) -> None:
    api = CoralAPI(base_url)
    print("TPU status:", api.status())
    print("\nclassify:")
    for r in api.classify(image, "tf2_mobilenet_v2_1.0_224_ptq_edgetpu", top_k=3)["results"]:
        print(f"  {r['score']:.3f}  {r['label'] or r['index']}")
    print("\ndetect:")
    for r in api.detect(image, "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu")["results"]:
        print(f"  {r['score']:.3f}  {r['label'] or r['index']}  box={r['box']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python examples/client.py <base_url> <image>")
        raise SystemExit(1)
    _demo(sys.argv[1], sys.argv[2])
