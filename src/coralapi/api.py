"""Versioned HTTP API (v1).

Interactive documentation is generated from these definitions and served at
/docs (Swagger UI) and /redoc; the raw schema is at /openapi.json.

/healthz and /readyz are intentionally unversioned: they are Kubernetes
probe endpoints, not part of the public API contract.
"""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response, UploadFile
from PIL import Image as PILImage

from coralapi import __version__, metrics
from coralapi.config import Settings
from coralapi.errors import classify_exception
from coralapi.runtime import device, queue, tflite
from coralapi.runtime.engine import COCO_KEYPOINTS
from coralapi.runtime.jobs import JobStore
from coralapi.runtime.registry import ModelRegistry
from coralapi.schemas import (
    ClassifyResponse,
    DetectResponse,
    EmbedResponse,
    JobStatus,
    Keypoint,
    ModelInfo,
    PoseResponse,
    QueueStatus,
    RawResponse,
    RuntimeStatus,
    SegmentClass,
    SegmentResponse,
)
from coralapi.uploads import open_image_checked, save_upload

router = APIRouter()

OPENAPI_TAGS = [
    {"name": "system", "description": "Service health, TPU status, and queue state."},
    {
        "name": "models",
        "description": "Model discovery. Models are selected per inference request "
        "by name; missing models are downloaded from the configured model source.",
    },
    {
        "name": "inference",
        "description": "Synchronous inference. The request waits for the TPU and "
        "returns the result; responds 429 when the TPU queue is full.",
    },
    {
        "name": "jobs",
        "description": "Asynchronous inference. Submitting returns 202 with a job "
        "id immediately; poll the job endpoint for the result. Jobs share the "
        "same TPU queue as synchronous requests.",
    },
    {
        "name": "raw",
        "description": "Generic inference for any model coralapi has no dedicated "
        "decoder for (segmentation, pose, embeddings, custom). Returns the model's "
        "raw output tensors for the client to decode.",
    },
]

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"description": "Unknown model (not local and not downloadable)"},
    413: {"description": "Upload exceeds the size or resolution limits"},
    429: {"description": "TPU queue is full; retry later"},
    502: {"description": "Model download from the model source failed"},
}


def _state(request: Request) -> tuple[Settings, ModelRegistry, queue.TPUExecutor, JobStore]:
    app = request.app
    return app.state.settings, app.state.registry, app.state.executor, app.state.jobs


def _map_inference_error(exc: Exception) -> HTTPException:
    mapped = classify_exception(exc)
    if mapped is None:
        raise exc
    status, message = mapped
    return HTTPException(status, message)


async def _prepare(
    request: Request, file: UploadFile
) -> tuple[Settings, ModelRegistry, queue.TPUExecutor, JobStore, Path, Any]:
    """Stream the upload to disk and validate it as an image."""
    settings, registry, executor, jobs = _state(request)
    path = await save_upload(file, settings.max_upload_bytes, settings.resolved_upload_dir)
    try:
        image = open_image_checked(path, settings.max_image_dim, settings.allowed_image_formats)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return settings, registry, executor, jobs, path, image


async def _run_sync(
    executor: queue.TPUExecutor, path: Path, operation: str, model: str, fn: Callable[[], Any]
) -> Any:
    start = time.monotonic()
    status = "error"
    try:
        result = await executor.submit(fn)
        status = "ok"
        return result
    except Exception as exc:
        http_exc = _map_inference_error(exc)  # re-raises unmapped errors as 500s
        status = str(http_exc.status_code)
        raise http_exc from exc
    finally:
        metrics.record_inference(
            operation, model, status, time.monotonic() - start if status == "ok" else None
        )
        path.unlink(missing_ok=True)


def _submit_job(
    executor: queue.TPUExecutor,
    jobs: JobStore,
    kind: str,
    model: str,
    path: Path,
    fn: Callable[[], Any],
) -> JobStatus:
    try:
        executor.reserve()
    except queue.QueueFull as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(429, str(exc)) from exc
    job = jobs.create(kind=kind, model=model)
    jobs.start(job, executor.run_reserved, fn, cleanup=lambda: path.unlink(missing_ok=True))
    return JobStatus.model_validate(job, from_attributes=True)


# -- system ----------------------------------------------------------------


@router.get("/healthz", tags=["system"], summary="Liveness probe")
def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/readyz", tags=["system"], summary="Readiness probe")
def readyz(request: Request) -> dict:
    """Ready only if the Coral device bound at startup is still present and
    unchanged. Goes 503 when the device is unplugged, or re-enumerated on a
    replug (different sysfs path); then the loaded delegate is stale and
    `POST /v1/reset` is needed to rebind."""
    _, registry, *_ = _state(request)
    current = {d.path for d in device.discover_all()}
    bound = getattr(request.app.state, "bound_devices", set())
    if not current:
        raise HTTPException(503, "No Coral Edge TPU detected")
    if current != bound:
        raise HTTPException(
            503, "Edge TPU changed since startup (replug?); POST /v1/reset to rebind"
        )
    # Report a count, not sysfs paths (no host topology disclosure).
    return {"status": "ok", "devices": len(current), "models_available": len(registry.list())}


@router.post(
    "/v1/reset",
    tags=["system"],
    summary="Rebind the Edge TPU after a device replug",
    description=(
        "Drops loaded models (releasing their delegates) and re-acquires the "
        "TPU, so a device that was unplugged and replugged is rebound without "
        "restarting the process. Runs on the TPU thread, after any in-flight "
        "inference. Returns 503 if no TPU is present after the rebind."
    ),
    responses={503: {"description": "No Coral TPU available after reset"}},
)
def reset(request: Request) -> dict:
    settings, registry, executor, _ = _state(request)

    def do_reset() -> int:
        cleared = registry.clear()
        # Re-acquire a delegate to confirm the (possibly re-enumerated) device
        # is bindable; discarded here, each model reloads its own on next use.
        tflite.load_edgetpu_delegate(settings.device, settings.edgetpu_lib)
        return cleared

    try:
        cleared = executor.run_sync(do_reset)
    except (tflite.InterpreterUnavailable, tflite.DelegateUnavailable) as exc:
        # Route through the sanitizer: log detail server-side, return a generic
        # message (don't leak str(exc) to the client).
        raise _map_inference_error(exc) from exc
    request.app.state.bound_devices = {d.path for d in device.discover_all()}
    return {
        "reset": True,
        "models_cleared": cleared,
        "devices": len(request.app.state.bound_devices),
    }


@router.get(
    "/v1/status",
    tags=["system"],
    summary="TPU runtime and hardware status",
    description="Reports the TFLite interpreter in use, the configured device "
    "preference, and the Edge TPU hardware discovered on the host.",
)
def status(request: Request) -> RuntimeStatus:
    settings, *_ = _state(request)
    return RuntimeStatus(
        interpreter=tflite.interpreter_source(),
        device_preference=settings.device,
        # A count, not sysfs paths (no host device-topology disclosure).
        devices=len(device.discover_all()),
    )


@router.get(
    "/v1/queue",
    tags=["system"],
    summary="Inference queue state",
)
def queue_status(request: Request) -> QueueStatus:
    _, _, executor, _ = _state(request)
    return QueueStatus(depth=executor.queue_depth, pending=executor.pending)


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus exposition. Mounted at /metrics only when metrics are enabled
    (see main.create_app); not part of the versioned API."""
    _, registry, executor, _ = _state(request)
    body, content_type = metrics.render(
        pending=executor.pending,
        depth=executor.queue_depth,
        models_loaded=registry.loaded_count,
    )
    return Response(content=body, media_type=content_type)


# -- models ------------------------------------------------------------------


@router.get(
    "/v1/models",
    tags=["models"],
    summary="List locally available models",
    description="Lists *.tflite files in the model directory. Inference "
    "endpoints also accept names of models not listed here; those are "
    "downloaded on first use when model download is enabled.",
)
def list_models(request: Request) -> list[ModelInfo]:
    _, registry, *_ = _state(request)
    return registry.list()


# -- inference (sync) ---------------------------------------------------------


@router.post(
    "/v1/vision/classify",
    tags=["inference"],
    summary="Classify an image (waits for result)",
    responses=_ERROR_RESPONSES,
)
async def classify(
    request: Request,
    file: UploadFile,
    model: str = Query(description="Model name, e.g. mobilenet_v2_1.0_224_quant_edgetpu"),
    top_k: int = Query(5, ge=1, le=100, description="Number of top classes to return"),
) -> ClassifyResponse:
    _, registry, executor, _, path, image = await _prepare(request, file)

    def fn() -> ClassifyResponse:
        results = registry.get(model).classify(image, top_k=top_k)
        return ClassifyResponse(model=model, results=results)

    return await _run_sync(executor, path, "classify", model, fn)


@router.post(
    "/v1/vision/detect",
    tags=["inference"],
    summary="Detect objects in an image (waits for result)",
    responses=_ERROR_RESPONSES,
)
async def detect(
    request: Request,
    file: UploadFile,
    model: str = Query(
        description="Model name, e.g. ssd_mobilenet_v2_coco_quant_postprocess_edgetpu"
    ),
    threshold: float = Query(0.4, ge=0.0, le=1.0, description="Minimum confidence score"),
) -> DetectResponse:
    _, registry, executor, _, path, image = await _prepare(request, file)

    def fn() -> DetectResponse:
        results = registry.get(model).detect(image, threshold=threshold)
        return DetectResponse(model=model, results=results)

    return await _run_sync(executor, path, "detect", model, fn)


def _segment_response(model: str, class_map: np.ndarray, labels: dict[int, str]) -> SegmentResponse:
    height, width = class_map.shape
    indices, counts = np.unique(class_map, return_counts=True)
    classes = sorted(
        (
            SegmentClass(index=int(i), label=labels.get(int(i)), pixels=int(c))
            for i, c in zip(indices, counts, strict=True)
        ),
        key=lambda c: c.pixels,
        reverse=True,
    )
    buf = io.BytesIO()
    PILImage.fromarray(class_map.astype(np.uint8), mode="L").save(buf, format="PNG")
    return SegmentResponse(
        model=model,
        width=int(width),
        height=int(height),
        classes=classes,
        mask_png=base64.b64encode(buf.getvalue()).decode(),
    )


@router.post(
    "/v1/vision/segment",
    tags=["inference"],
    summary="Semantic segmentation (per-pixel class mask)",
    description=(
        "Runs a segmentation model (e.g. DeepLab) and returns a per-pixel class "
        "map as a base64 PNG (mode L, pixel value = class index) plus a summary "
        "of the classes present. Labels come from the model's labels sidecar if "
        "present."
    ),
    responses=_ERROR_RESPONSES,
)
async def segment(
    request: Request,
    file: UploadFile,
    model: str = Query(description="Model name, e.g. deeplabv3_mnv2_pascal_quant_edgetpu"),
) -> SegmentResponse:
    _, registry, executor, _, path, image = await _prepare(request, file)

    def fn() -> SegmentResponse:
        loaded = registry.get(model)
        return _segment_response(model, loaded.segment(image), loaded.labels)

    return await _run_sync(executor, path, "segment", model, fn)


@router.post(
    "/v1/vision/pose",
    tags=["inference"],
    summary="Single-pose keypoint estimation",
    description=(
        "Runs a single-pose model (e.g. MoveNet) and returns keypoints as named "
        "(x, y, score) points, normalized to 0..1. 17-keypoint models are named "
        "in COCO order; others are named kp0, kp1, …"
    ),
    responses=_ERROR_RESPONSES,
)
async def pose(
    request: Request,
    file: UploadFile,
    model: str = Query(description="Model name, e.g. movenet_single_pose_lightning_ptq_edgetpu"),
) -> PoseResponse:
    _, registry, executor, _, path, image = await _prepare(request, file)

    def fn() -> PoseResponse:
        kps = registry.get(model).pose(image)
        names = COCO_KEYPOINTS if len(kps) == len(COCO_KEYPOINTS) else None
        keypoints = [
            Keypoint(
                name=names[i] if names else f"kp{i}",
                x=float(x),
                y=float(y),
                score=float(score),
            )
            for i, (y, x, score) in enumerate(kps)
        ]
        return PoseResponse(model=model, keypoints=keypoints)

    return await _run_sync(executor, path, "pose", model, fn)


@router.post(
    "/v1/vision/embed",
    tags=["inference"],
    summary="Image embedding / feature vector",
    description=(
        "Runs an embedding-extractor model and returns its output feature vector, "
        "for similarity search, deduplication, or on-device imprinting. Pass "
        "normalize=true for an L2-normalized vector (cosine similarity = dot product)."
    ),
    responses=_ERROR_RESPONSES,
)
async def embed(
    request: Request,
    file: UploadFile,
    model: str = Query(
        description="Model name, e.g. mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu"
    ),
    normalize: bool = Query(False, description="L2-normalize the returned vector"),
) -> EmbedResponse:
    _, registry, executor, _, path, image = await _prepare(request, file)

    def fn() -> EmbedResponse:
        vector = registry.get(model).embed(image, normalize=normalize)
        return EmbedResponse(
            model=model,
            dimension=len(vector),
            normalized=normalize,
            embedding=[float(v) for v in vector],
        )

    return await _run_sync(executor, path, "embed", model, fn)


# -- raw (generic) ------------------------------------------------------------


@router.post(
    "/v1/raw/infer",
    tags=["raw"],
    summary="Run any model and return its raw output tensors",
    description=(
        "Uploads an image, resizes it to the model's input tensor, runs inference, "
        "and returns **every output tensor** with its name, shape, dtype, and "
        "quantization — no task-specific decoding. Use this for models coralapi "
        "has no dedicated endpoint for (semantic segmentation, pose estimation, "
        "embedding extractors, or any custom `.tflite`), and decode the tensors "
        "client-side using the `shape` metadata. Integer outputs are dequantized "
        "to float32 by default (`dequantize=false` returns the raw quantized "
        "values plus the scale/zero_point). Note: output tensors can be large "
        "(a segmentation map is width x height x classes)."
    ),
    responses=_ERROR_RESPONSES,
)
async def raw_infer(
    request: Request,
    file: UploadFile,
    model: str = Query(description="Model name (downloaded on first use if not local)"),
    dequantize: bool = Query(
        True, description="Dequantize integer outputs to float32 (else return raw values)"
    ),
) -> RawResponse:
    _, registry, executor, _, path, image = await _prepare(request, file)

    def fn() -> RawResponse:
        loaded = registry.get(model)
        return RawResponse(
            model=model,
            input_shape=[int(x) for x in loaded.input_detail["shape"]],
            outputs=loaded.raw(image, dequantize=dequantize),
        )

    return await _run_sync(executor, path, "raw", model, fn)


# -- inference (async jobs) ---------------------------------------------------


@router.post(
    "/v1/jobs/classify",
    tags=["jobs"],
    status_code=202,
    summary="Submit an async classification job",
    responses=_ERROR_RESPONSES,
)
async def submit_classify(
    request: Request,
    file: UploadFile,
    model: str = Query(description="Model name"),
    top_k: int = Query(5, ge=1, le=100),
) -> JobStatus:
    _, registry, executor, jobs, path, image = await _prepare(request, file)

    def fn() -> ClassifyResponse:
        results = registry.get(model).classify(image, top_k=top_k)
        return ClassifyResponse(model=model, results=results)

    return _submit_job(executor, jobs, "classify", model, path, fn)


@router.post(
    "/v1/jobs/detect",
    tags=["jobs"],
    status_code=202,
    summary="Submit an async detection job",
    responses=_ERROR_RESPONSES,
)
async def submit_detect(
    request: Request,
    file: UploadFile,
    model: str = Query(description="Model name"),
    threshold: float = Query(0.4, ge=0.0, le=1.0),
) -> JobStatus:
    _, registry, executor, jobs, path, image = await _prepare(request, file)

    def fn() -> DetectResponse:
        results = registry.get(model).detect(image, threshold=threshold)
        return DetectResponse(model=model, results=results)

    return _submit_job(executor, jobs, "detect", model, path, fn)


@router.get(
    "/v1/jobs/{job_id}",
    tags=["jobs"],
    summary="Poll an async job",
    responses={404: {"description": "Unknown or expired job id"}},
)
def get_job(request: Request, job_id: str) -> JobStatus:
    _, _, _, jobs = _state(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown or expired job {job_id!r}")
    return JobStatus.model_validate(job, from_attributes=True)
