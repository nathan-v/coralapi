import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from coralapi import __version__, api, metrics
from coralapi.config import Settings, settings
from coralapi.middleware import BodySizeLimitMiddleware
from coralapi.runtime import device, tflite
from coralapi.runtime.jobs import JobStore
from coralapi.runtime.queue import TPUExecutor
from coralapi.runtime.registry import ModelRegistry

logger = logging.getLogger("coralapi")

DESCRIPTION = """\
Inference API for Google Coral Edge TPUs (PCIe and USB).

Models are selected per request by name; models not present locally are
downloaded from the configured model source. The TPU executes one inference
at a time — concurrent requests queue up to a configurable depth, then are
rejected with 429. Both synchronous requests and asynchronous jobs are
supported.
"""


def require_edgetpu(config: Settings, executor: TPUExecutor) -> None:
    """Verify a Coral TPU is present, or abort startup with an actionable error.

    The probe runs on the executor's dedicated TPU thread — the same thread all
    later inference runs on — because libedgetpu delegates are thread-affine.
    There is deliberately no CPU fallback: a running coralapi instance always
    means a working TPU.
    """

    def probe() -> None:
        # Load a delegate and discard it — this only proves the device is
        # present and usable on the TPU thread. Each model loads its own.
        tflite.load_edgetpu_delegate(config.device, config.edgetpu_lib)

    try:
        executor.run_sync(probe)
    except (tflite.InterpreterUnavailable, tflite.DelegateUnavailable) as exc:
        found = device.discover_all()
        raise RuntimeError(
            f"Coral Edge TPU unavailable (device preference {config.device!r}): {exc}\n"
            f"Hardware discovered on this host: "
            f"{[d.path for d in found] or 'none'}.\n"
            "Check that libedgetpu is installed and the device is attached and "
            "passed through to the container "
            "(--device /dev/apex_0 for PCIe, --device /dev/bus/usb for USB)."
        ) from exc
    logger.info(
        "Edge TPU verified (preference=%s, hardware=%s)",
        config.device,
        [d.path for d in device.discover_all()],
    )


def create_app(config: Settings | None = None) -> FastAPI:
    cfg = config or settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail startup outright if no Coral TPU is available (probe runs on the
        # dedicated TPU thread).
        require_edgetpu(cfg, app.state.executor)
        # Record which device(s) the delegate is bound to, so /readyz can detect
        # an unplug/replug and /v1/reset can rebind.
        app.state.bound_devices = {d.path for d in device.discover_all()}
        _prepare_upload_dir(cfg.resolved_upload_dir)
        yield
        app.state.executor.shutdown()

    app = FastAPI(
        title="coralapi",
        description=DESCRIPTION,
        version=__version__,
        openapi_tags=api.OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.registry = ModelRegistry(
        model_dir=cfg.model_dir,
        model_source=cfg.model_source,
        model_download=cfg.model_download,
        allow_insecure_source=cfg.allow_insecure_model_source,
        max_downloaded_models=cfg.max_downloaded_models,
        device=cfg.device,
        edgetpu_lib=cfg.edgetpu_lib,
    )
    app.state.executor = TPUExecutor(cfg.queue_depth)
    app.state.jobs = JobStore(cfg.job_retention_seconds)
    # Bound unauthenticated resource use on the upload path at the ASGI layer,
    # before any body reaches disk (see middleware.py).
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=cfg.max_upload_bytes)
    app.include_router(api.router)
    if cfg.metrics_enabled:
        if not metrics.AVAILABLE:
            raise RuntimeError(
                "CORALAPI_METRICS_ENABLED is set but prometheus-client is not "
                "installed. Install the 'prometheus' extra."
            )
        app.add_api_route(
            "/metrics",
            api.metrics_endpoint,
            methods=["GET"],
            tags=["system"],
            summary="Prometheus metrics exposition",
        )
    return app


def _prepare_upload_dir(upload_dir: Path) -> None:
    """Create the upload dir (private perms) and sweep orphaned uploads.

    A hard crash can orphan a streamed upload; clearing on startup keeps a
    restart loop from accumulating disk. Files are named coralapi-upload-*.
    """
    upload_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for stale in upload_dir.glob("coralapi-upload-*"):
        try:
            stale.unlink()
        except OSError:
            logger.warning("Could not remove stale upload %s", stale)


app = create_app()
