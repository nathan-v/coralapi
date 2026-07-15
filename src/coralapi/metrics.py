"""Optional Prometheus instrumentation.

`prometheus-client` is an optional dependency (the `prometheus` extra). When it
is not installed, every function here is a no-op, so the inference paths can
call them unconditionally with no import cost. The `/metrics` endpoint is only
mounted when CORALAPI_METRICS_ENABLED is set (see main.create_app), which also
requires the dependency to be present.
"""

from __future__ import annotations

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    AVAILABLE = False

if AVAILABLE:
    _INFERENCES = Counter(
        "coralapi_inferences_total",
        "Inference requests processed, by operation, model, and outcome.",
        ["operation", "model", "status"],
    )
    _LATENCY = Histogram(
        "coralapi_inference_duration_seconds",
        "Successful inference latency (TPU submit to result), by operation and model.",
        ["operation", "model"],
    )
    _QUEUE_PENDING = Gauge(
        "coralapi_queue_pending", "Requests currently running or waiting for the TPU."
    )
    _QUEUE_DEPTH = Gauge("coralapi_queue_depth", "Configured max requests allowed to wait.")
    _MODELS_LOADED = Gauge("coralapi_models_loaded", "Models currently loaded on the TPU.")


def record_inference(operation: str, model: str, status: str, duration: float | None) -> None:
    """Count one inference; record its latency only on success (status 'ok')."""
    if not AVAILABLE:
        return
    _INFERENCES.labels(operation, model, status).inc()
    if status == "ok" and duration is not None:
        _LATENCY.labels(operation, model).observe(duration)


def render(pending: int, depth: int, models_loaded: int) -> tuple[bytes, str]:
    """Set scrape-time gauges and return (body, content_type). Requires AVAILABLE."""
    _QUEUE_PENDING.set(pending)
    _QUEUE_DEPTH.set(depth)
    _MODELS_LOADED.set(models_loaded)
    return generate_latest(), CONTENT_TYPE_LATEST
