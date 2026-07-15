import re
import tempfile
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Accepted CORALAPI_DEVICE values: auto | usb | pci | <usb|pci|>:<index>
_DEVICE_RE = re.compile(r"^(auto|usb|pci|(usb|pci)?:\d+)$")


class Settings(BaseSettings):
    """Runtime configuration, overridable via CORALAPI_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="CORALAPI_")

    # Directory scanned for *.tflite models (mount your model volume here).
    model_dir: Path = Path("/models")

    # Which Edge TPU to bind. There is deliberately no CPU fallback: the
    # service refuses to start if no Coral TPU is available.
    #   auto    - first available Edge TPU
    #   usb     - first USB Accelerator
    #   pci     - first PCIe/M.2 module (/dev/apex_*)
    #   pci:0, pci:1, usb:0, :1 - a SPECIFIC device by index. Use this to pin
    #             one process to one TPU on a multi-TPU host (e.g. a dual-PCIe
    #             card): run one replica per TPU, each with a different index.
    device: str = "auto"

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not _DEVICE_RE.match(value):
            raise ValueError(
                f"invalid device {value!r}; expected auto, usb, pci, or a "
                "specific index like pci:0 / usb:1 / :0"
            )
        return value

    # Override the delegate shared library path if libedgetpu is installed
    # somewhere non-standard.
    edgetpu_lib: str | None = None

    # Base URL that missing models are downloaded from as <base>/<name>.tflite.
    # Defaults to the Coral model zoo's test_data mirror. Set model_download
    # to false to serve only models already present in model_dir. Must be
    # https unless allow_insecure_model_source is set (downloads are executed
    # server-side and loaded onto the TPU, so integrity matters).
    model_source: str = "https://raw.githubusercontent.com/google-coral/test_data/master"
    model_download: bool = True
    allow_insecure_model_source: bool = False

    # Cap on how many models a client can cause to be downloaded into
    # model_dir, bounding disk use on the attacker-influenced download path.
    max_downloaded_models: int = 100

    # Number of inference requests allowed to wait for the TPU (which executes
    # one inference at a time) before new requests are rejected with 429.
    queue_depth: int = 16

    # Expose Prometheus metrics at GET /metrics. Off by default; requires the
    # 'prometheus' extra (prometheus-client) to be installed when enabled.
    metrics_enabled: bool = False

    # How long completed async job results are kept for polling.
    job_retention_seconds: float = 3600

    # Uploads are streamed to a dedicated, per-instance directory (defaults to
    # <model_dir sibling>/uploads via resolved_upload_dir) and rejected once
    # they exceed max_upload_bytes. The limit is enforced at the ASGI layer
    # (BodySizeLimitMiddleware) before any body reaches disk.
    upload_dir: Path | None = None
    max_upload_bytes: int = 500 * 1024 * 1024  # 500 MB

    @property
    def resolved_upload_dir(self) -> Path:
        """Concrete upload directory; isolated from system temp by default."""
        return self.upload_dir or (Path(tempfile.gettempdir()) / "coralapi-uploads")

    # Maximum input resolution, as the longest allowed edge in pixels.
    # 15360 px is 16K. max_video_dim is reserved for the planned video
    # endpoints and enforced the same way.
    max_image_dim: int = 15360
    max_video_dim: int = 15360


settings = Settings()
