"""Model execution: preprocessing, invocation, and output parsing.

A LoadedModel wraps one TFLite interpreter bound to its own Edge TPU
delegate (delegates are not shared — see runtime/queue.py). There is no CPU
execution path: the service refuses to start without a TPU, so every
interpreter here runs delegated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from coralapi.runtime import tflite
from coralapi.schemas import Classification, Detection


def load_labels(path: Path) -> dict[int, str]:
    """Parse a labels file: either 'label' per line or 'index label' per line."""
    labels: dict[int, str] = {}
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        first, _, rest = line.partition(" ")
        if first.isdigit() and rest:
            labels[int(first)] = rest.strip()
        else:
            labels[i] = line
    return labels


# COCO 17-keypoint order used by MoveNet / PoseNet single-pose models.
COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class LoadedModel:
    def __init__(
        self,
        name: str,
        path: Path,
        device: str = "auto",
        edgetpu_lib: str | None = None,
        labels: dict[int, str] | None = None,
    ):
        self.name = name
        self.path = path
        self.labels = labels or {}
        # Each model owns its own delegate, created on whatever thread this
        # runs on (the TPUExecutor's dedicated thread). Sharing one delegate
        # across interpreters, or using it from another thread, fails at invoke
        # ("EdgeTpuDelegateForCustomOp failed to invoke"). Kept as an attribute
        # so it outlives __init__ (the interpreter holds only a borrowed ref).
        self._delegate = tflite.load_edgetpu_delegate(device, edgetpu_lib)
        self.interpreter = tflite.make_interpreter(str(path), [self._delegate])
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

    # -- input handling --------------------------------------------------

    @property
    def _nchw(self) -> bool:
        """True if the model wants channels-first input `[1, 3, H, W]` (some
        ultralytics exports) rather than the usual NHWC `[1, H, W, 3]`."""
        shape = self.input_detail["shape"]
        return len(shape) == 4 and int(shape[1]) == 3 and int(shape[3]) != 3

    @property
    def input_size(self) -> tuple[int, int]:
        """(width, height) expected by the model."""
        shape = self.input_detail["shape"]
        if self._nchw:
            return int(shape[3]), int(shape[2])  # NCHW [1, 3, H, W]
        return int(shape[2]), int(shape[1])  # NHWC [1, H, W, 3]

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(self.input_size, Image.LANCZOS)
        data = np.asarray(image)  # HWC uint8, 0..255
        dtype = self.input_detail["dtype"]
        if dtype == np.float32:
            data = data.astype(np.float32) / 255.0
        elif dtype == np.int8:
            # int8-input exports (e.g. ultralytics YOLO) need real quantization:
            # a plain astype(int8) wraps pixels >127 to negative and scrambles the
            # image. Shift into the model's int8 domain via its quant params (for
            # the common scale=1/255, zero_point=-128 this reduces to pixels-128).
            scale, zero_point = self.input_detail["quantization"]
            quantized = np.round(data.astype(np.float32) / 255.0 / scale + zero_point)
            data = np.clip(quantized, -128, 127).astype(np.int8)
        else:  # uint8 (Coral zoo models): already in range, astype is identity
            data = data.astype(dtype)
        tensor = np.expand_dims(data, axis=0)  # NHWC [1, H, W, C]
        if self._nchw:
            tensor = np.transpose(tensor, (0, 3, 1, 2))  # -> NCHW [1, C, H, W]
        return tensor

    def _invoke(self, image: Image.Image) -> list[np.ndarray]:
        self.interpreter.set_tensor(self.input_detail["index"], self._preprocess(image))
        self.interpreter.invoke()
        return [self._dequantize(d) for d in self.output_details]

    def _dequantize(self, detail: dict) -> np.ndarray:
        data = self.interpreter.get_tensor(detail["index"])
        if data.dtype in (np.uint8, np.int8):
            scale, zero_point = detail.get("quantization", (0.0, 0))
            if scale:
                return (data.astype(np.float32) - zero_point) * scale
        return data.astype(np.float32)

    # -- generic (no task-specific decoding) -------------------------------

    def raw(self, image: Image.Image, dequantize: bool = True) -> list[dict]:
        """Run inference and return every output tensor with its metadata.

        The escape hatch for models coralapi has no dedicated decoder for
        (segmentation, pose, embeddings, or anything custom): the client gets
        the tensors and decodes them itself. Integer outputs are dequantized to
        float32 by default; pass dequantize=False for the raw quantized values.
        """
        self.interpreter.set_tensor(self.input_detail["index"], self._preprocess(image))
        self.interpreter.invoke()
        outputs: list[dict] = []
        for detail in self.output_details:
            arr = self.interpreter.get_tensor(detail["index"])
            scale, zero_point = detail.get("quantization", (0.0, 0))
            quant = {"scale": float(scale), "zero_point": int(zero_point)} if scale else None
            if dequantize and quant and arr.dtype in (np.uint8, np.int8):
                values = (arr.astype(np.float32) - zero_point) * scale
                dtype = "float32"
            else:
                values = arr
                dtype = str(arr.dtype)
            outputs.append(
                {
                    "name": str(detail["name"]),
                    "shape": [int(x) for x in arr.shape],
                    "dtype": dtype,
                    "quantization": quant,
                    "data": values.ravel().tolist(),
                }
            )
        return outputs

    def segment(self, image: Image.Image) -> np.ndarray:
        """Return the per-pixel class-index map as an (H, W) int array.

        Handles both DeepLab-style outputs: a pre-argmaxed `(1, H, W)` class map,
        and `(1, H, W, C)` per-class logits (argmaxed here).
        """
        out = self._invoke(image)[0]
        a = out[0] if out.ndim >= 3 else out  # drop the batch dimension
        if a.ndim == 3:  # (H, W, C) logits
            a = a.argmax(axis=-1)
        return a.astype(np.int64)

    def pose(self, image: Image.Image) -> np.ndarray:
        """Return keypoints as a (K, 3) array of (y, x, score), normalized 0..1.

        Matches the MoveNet / PoseNet single-pose output `(1, 1, K, 3)`.
        """
        return self._invoke(image)[0].reshape(-1, 3)

    def embed(self, image: Image.Image, normalize: bool = False) -> np.ndarray:
        """Return the model's output feature vector as a 1-D float array."""
        vector = self._invoke(image)[0].reshape(-1).astype(np.float32)
        if normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        return vector

    # -- task heads --------------------------------------------------------

    def classify(self, image: Image.Image, top_k: int = 5) -> list[Classification]:
        scores = self._invoke(image)[0].squeeze()
        top = np.argsort(scores)[::-1][:top_k]
        return [
            Classification(index=int(i), label=self.labels.get(int(i)), score=float(scores[i]))
            for i in top
        ]

    def detect(self, image: Image.Image, threshold: float = 0.4) -> list[Detection]:
        """Run detection, dispatching on the model's output head.

        Supported heads: the standard 4-tensor TFLite SSD postprocess
        (boxes/classes/scores/count), and the single-tensor raw head that
        Ultralytics YOLO Edge TPU exports produce (decoded + NMS here).
        """
        outputs = self._invoke(image)
        if len(outputs) >= 4:
            return decode_ssd_outputs(outputs, self.labels, threshold)
        if len(outputs) == 1 and outputs[0].ndim == 3:
            return decode_yolo_output(outputs[0], self.labels, threshold)
        raise ValueError(
            f"Model {self.name!r} has an unsupported detection head "
            f"({len(outputs)} outputs); expected the 4-tensor SSD postprocess "
            "or a single-tensor YOLO head."
        )


def decode_ssd_outputs(
    outputs: list[np.ndarray], labels: dict[int, str], threshold: float
) -> list[Detection]:
    """Parse the standard TFLite SSD postprocess outputs.

    The head emits four tensors — boxes, class ids, scores, count — but their
    order is not consistent across exporters: the TF1 TFLite_Detection_PostProcess
    family uses (boxes, classes, scores, count) while the TF2 Object Detection
    API exports use (boxes, scores, classes, count). Boxes and count are pinned
    by shape; classes vs scores are told apart by value (scores are a
    probability in [0, 1]; class ids are integer-valued), so both orderings
    decode correctly.
    """
    boxes = next(o for o in outputs if o.ndim == 3 and o.shape[-1] == 4)
    # .item() rather than int(...): int() on a size-1 ndarray raises under numpy 2.x.
    count = int(next(o for o in outputs if o.size == 1).item())
    flat = [o.squeeze(0) for o in outputs if o.ndim == 2]
    classes, scores = _split_classes_scores(flat[0], flat[1], count)

    detections = []
    for i in range(min(count, len(scores))):
        if scores[i] < threshold:
            continue
        idx = int(classes[i])
        detections.append(
            Detection(
                box=[float(v) for v in boxes[0][i]],
                index=idx,
                label=labels.get(idx),
                score=float(scores[i]),
            )
        )
    return detections


def _split_classes_scores(
    a: np.ndarray, b: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Given the two (N,) postprocess tensors, return (classes, scores).

    Decides by value, not position: a scores tensor is a probability bounded
    by [0, 1]; a class-id tensor is integer-valued and, for any multi-class
    model, ranges past 1. Only the first `count` entries are examined because
    padding rows beyond the detection count can be uninitialized (NaN/garbage).
    Falls back to the TF1 order (a=classes) only when the two are genuinely
    indistinguishable (e.g. a binary model whose scores are all exactly 0/1).
    """

    def looks_integral(v: np.ndarray) -> bool:
        finite = v[np.isfinite(v)]
        return finite.size == 0 or bool(np.allclose(finite, np.round(finite)))

    def max_finite(v: np.ndarray) -> float:
        finite = v[np.isfinite(v)]
        return float(finite.max()) if finite.size else 0.0

    n = count if count > 0 else min(len(a), len(b))
    va, vb = a[:n], b[:n]
    a_over_one, b_over_one = max_finite(va) > 1.01, max_finite(vb) > 1.01

    # A value above 1 can only be a class id, never a probability.
    if a_over_one and not b_over_one:
        return a, b
    if b_over_one and not a_over_one:
        return b, a

    # Both within [0, 1]: the integer-valued tensor is the class ids.
    a_int, b_int = looks_integral(va), looks_integral(vb)
    if b_int and not a_int:
        return b, a
    if a_int and not b_int:
        return a, b

    return a, b  # indistinguishable — assume TF1 (boxes, classes, scores, count)


def decode_yolo_output(
    output: np.ndarray,
    labels: dict[int, str],
    threshold: float,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Decode an Ultralytics YOLO (v8/11/26) Edge TPU export head.

    The export emits one tensor of shape (1, 4 + num_classes, num_anchors)
    holding normalized center-x/center-y/width/height rows followed by
    per-class scores; there is no built-in NMS, so it happens here.
    """
    arr = output.squeeze(0)
    if arr.shape[0] > arr.shape[1]:  # tolerate (anchors, channels) layout
        arr = arr.T
    xywh = arr[:4]
    class_scores = arr[4:]
    class_ids = class_scores.argmax(axis=0)
    scores = class_scores.max(axis=0)

    keep = scores >= threshold
    if not keep.any():
        return []
    xywh, class_ids, scores = xywh[:, keep], class_ids[keep], scores[keep]

    cx, cy, w, h = xywh
    boxes = np.stack([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2], axis=1)

    # Class-aware NMS via the coordinate-offset trick (boxes are in 0..1).
    order = np.argsort(scores)[::-1]
    offset_boxes = boxes + class_ids[:, None].astype(np.float32) * 2.0
    kept: list[int] = []
    while order.size:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        iou = _box_iou(offset_boxes[i], offset_boxes[rest])
        order = rest[iou <= iou_threshold]

    return [
        Detection(
            box=[float(v) for v in boxes[i]],
            index=int(class_ids[i]),
            label=labels.get(int(class_ids[i])),
            score=float(scores[i]),
        )
        for i in kept
    ]


def _box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one [ymin, xmin, ymax, xmax] box against a (N, 4) array."""
    ymin = np.maximum(box[0], others[:, 0])
    xmin = np.maximum(box[1], others[:, 1])
    ymax = np.minimum(box[2], others[:, 2])
    xmax = np.minimum(box[3], others[:, 3])
    intersection = np.clip(ymax - ymin, 0, None) * np.clip(xmax - xmin, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return intersection / (area + other_areas - intersection + 1e-9)
