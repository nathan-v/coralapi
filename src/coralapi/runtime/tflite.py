"""Locate a TFLite interpreter implementation and the Edge TPU delegate.

We deliberately avoid pycoral: it is unmaintained and pinned to Python 3.9.
The same acceleration is available by handing the libedgetpu delegate to a
plain TFLite interpreter, which keeps us on current Python and current
tflite-runtime releases.

All imports here are lazy and optional so the service can run (and tests can
pass) on machines with no TPU stack installed at all.
"""

from __future__ import annotations

import platform
from functools import cache
from typing import Any

EDGETPU_LIB_DEFAULTS = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}


class InterpreterUnavailable(RuntimeError):
    """No TFLite interpreter implementation is installed."""


class DelegateUnavailable(RuntimeError):
    """libedgetpu could not be loaded or no Edge TPU matched the request."""


@cache
def interpreter_module() -> Any | None:
    """Return the module exposing Interpreter/load_delegate, or None."""
    try:
        from tflite_runtime import interpreter  # type: ignore[import-not-found]

        return interpreter
    except ImportError:
        pass
    try:
        # LiteRT is Google's successor package to tflite-runtime.
        from ai_edge_litert import interpreter  # type: ignore[import-not-found]

        return interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf  # type: ignore[import-not-found]

        return tf.lite
    except ImportError:
        return None


def interpreter_source() -> str | None:
    mod = interpreter_module()
    return getattr(mod, "__name__", None) if mod is not None else None


def _load_delegate_fn(mod: Any):
    fn = getattr(mod, "load_delegate", None)
    if fn is None:
        # tensorflow keeps it under tf.lite.experimental
        fn = getattr(getattr(mod, "experimental", None), "load_delegate", None)
    return fn


def load_edgetpu_delegate(device: str, lib_override: str | None = None) -> Any:
    """Load the Edge TPU delegate bound to a device preference.

    device: "auto" (first available), a type ("usb"/"pci"), or a specific
    device by index ("pci:0", "pci:1", "usb:0", ":1") to pin to one TPU on a
    multi-TPU host. The value is passed through to libedgetpu's device option.
    """
    mod = interpreter_module()
    if mod is None:
        raise InterpreterUnavailable(
            "No TFLite interpreter installed. Install the 'tpu' extra "
            "(tflite-runtime) or tensorflow."
        )
    load_delegate = _load_delegate_fn(mod)
    if load_delegate is None:
        raise DelegateUnavailable("Interpreter implementation has no load_delegate()")

    lib = lib_override or EDGETPU_LIB_DEFAULTS.get(platform.system(), "libedgetpu.so.1")
    options = {} if device == "auto" else {"device": device}
    try:
        return load_delegate(lib, options)
    except (ValueError, OSError) as exc:
        raise DelegateUnavailable(f"Could not load Edge TPU delegate {lib!r}: {exc}") from exc


def make_interpreter(model_path: str, delegates: list[Any]) -> Any:
    mod = interpreter_module()
    if mod is None:
        raise InterpreterUnavailable(
            "No TFLite interpreter installed. Install the 'tpu' extra "
            "(tflite-runtime) or tensorflow."
        )
    return mod.Interpreter(model_path=model_path, experimental_delegates=delegates)
