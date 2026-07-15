# Examples

[`client.py`](client.py) — a small, dependency-light Python client for CoralAPI, with a method per endpoint and decoding helpers (e.g. `segment_mask` returns a NumPy array). Import it, or run it directly for a quick demo:

```sh
python examples/client.py http://localhost:8000 cat.jpg
```

```python
from examples.client import CoralAPI

api = CoralAPI("http://localhost:8000")
api.classify("cat.jpg", "tf2_mobilenet_v2_1.0_224_ptq_edgetpu", top_k=3)
api.detect("street.jpg", "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu")
mask = api.segment_mask("street.jpg", "deeplabv3_mnv2_pascal_quant_edgetpu")   # (H, W) class-index array
vec = api.embed("product.jpg", "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu", normalize=True)
```

Needs `httpx` (and `numpy` + `pillow` for `segment_mask`) — all installed by `uv sync`.
