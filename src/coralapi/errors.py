"""Central mapping from internal exceptions to client-safe responses.

Both the synchronous request path (api.py) and the async job runner
(runtime/jobs.py) route failures through here so that internal detail —
download URLs, filesystem paths, library error strings — is logged
server-side but never returned to unauthenticated clients.
"""

from __future__ import annotations

import logging

from coralapi.runtime import queue, tflite
from coralapi.runtime.registry import ModelDownloadError, ModelNotFound

logger = logging.getLogger("coralapi")


def classify_exception(exc: BaseException) -> tuple[int, str] | None:
    """(status, client-safe message) for a known inference error, else None.

    The full exception is logged here; the returned message is deliberately
    generic for anything that could carry internal detail.
    """
    if isinstance(exc, ModelNotFound):
        # The name came from the client, so echoing it leaks nothing.
        return 404, f"Unknown model {exc.args[0]!r}"
    if isinstance(exc, queue.QueueFull):
        return 429, str(exc)
    if isinstance(exc, ModelDownloadError):
        logger.warning("Model download failed: %s", exc)
        return 502, "Model could not be retrieved from the model source"
    if isinstance(exc, tflite.InterpreterUnavailable | tflite.DelegateUnavailable):
        logger.error("TPU runtime error during inference: %s", exc)
        return 503, "Inference runtime unavailable"
    return None


def public_error_message(exc: BaseException) -> str:
    """Client-safe failure string for an async job's `error` field."""
    mapped = classify_exception(exc)
    if mapped is not None:
        return mapped[1]
    logger.exception("Unexpected inference job failure", exc_info=exc)
    return "Inference failed"
