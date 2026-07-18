"""Discover, download, and lazily load *.tflite models.

Conventions (matching coral.ai model zoo downloads):
  - a model is any <name>.tflite file directly inside the model directory
  - Edge TPU-compiled models end in _edgetpu.tflite
  - labels live in a sidecar file <name>.labels.txt (with <name>.txt and a
    shared labels.txt as fallbacks)

A model requested by name that is not present locally is fetched from the
configured model source as <source>/<name>.tflite and cached in model_dir.
"""

from __future__ import annotations

import hashlib
import re
import threading
import urllib.parse
from pathlib import Path

import httpx

from coralapi.runtime.engine import LoadedModel, load_labels
from coralapi.schemas import ModelInfo

_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
# Label files are tiny; a huge one is suspicious, so cap them tightly.
_LABEL_MAX_BYTES = 16 * 1024 * 1024


class ModelNotFound(KeyError):
    pass


class ModelDownloadError(RuntimeError):
    pass


def zoo_label_file(name: str) -> str | None:
    """Best-guess the shared label file a google-coral/test_data model uses, by
    name, so labels can be auto-fetched and `label` is non-null out of the box.

    Returns a filename present in the zoo (e.g. "coco_labels.txt"), or None for
    models with no class labels (face/pose/embeddings) or unrecognized names.
    A user-supplied `<name>.labels.txt` always takes precedence over this guess.
    """
    n = name.lower()
    # Most specific first — these substrings can co-occur with the generic ones.
    if "inat_bird" in n:
        return "inat_bird_labels.txt"
    if "inat_insect" in n:
        return "inat_insect_labels.txt"
    if "inat_plant" in n:
        return "inat_plant_labels.txt"
    if "cityscapes" in n:
        return "cityscapes_segmentation_labels.txt"
    if "deeplab" in n or "unet" in n or "pascal" in n:
        return "pascal_voc_segmentation_labels.txt"
    if "pet" in n:
        return "pet_labels.txt"
    if "face" in n or "pose" in n or "movenet" in n or "embedding" in n:
        return None  # single-class / keypoints / vectors — no label set
    if any(k in n for k in ("ssd", "efficientdet", "mobiledet", "yolo", "_coco", "fpn")):
        return "coco_labels.txt"
    if any(k in n for k in ("mobilenet", "inception", "efficientnet", "resnet", "imagenet")):
        return "imagenet_labels.txt"
    return None


class ModelRegistry:
    def __init__(
        self,
        model_dir: Path,
        model_source: str | None = None,
        model_download: bool = True,
        allow_insecure_source: bool = False,
        max_downloaded_models: int = 100,
        max_model_bytes: int = 256 * 1024 * 1024,
        max_model_cache_bytes: int = 4 * 1024 * 1024 * 1024,
        checksums: dict[str, str] | None = None,
        device: str = "auto",
        edgetpu_lib: str | None = None,
    ):
        self.model_dir = model_dir
        self.model_source = (model_source or "").rstrip("/")
        self.model_download = model_download and bool(self.model_source)
        self.max_downloaded_models = max_downloaded_models
        self.max_model_bytes = max_model_bytes
        self.max_model_cache_bytes = max_model_cache_bytes
        self.checksums = {k: v.lower() for k, v in (checksums or {}).items()}
        self.device = device
        self.edgetpu_lib = edgetpu_lib
        if self.model_download and not allow_insecure_source:
            scheme = urllib.parse.urlsplit(self.model_source).scheme
            if scheme != "https":
                raise ValueError(
                    f"model_source must be https (got {scheme!r}); set "
                    "allow_insecure_model_source to override."
                )
        # Host the download must stay on: redirects to any other host are
        # refused, so a redirecting source cannot turn into an SSRF primitive.
        self._source_host = urllib.parse.urlsplit(self.model_source).netloc
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    @property
    def loaded_count(self) -> int:
        """Number of models currently held loaded on the TPU."""
        return len(self._loaded)

    def clear(self) -> int:
        """Drop all loaded models, releasing their Edge TPU delegates; returns the
        count dropped. Call this on the TPU thread (delegate teardown is
        thread-affine, same as creation). The next `get()` reloads fresh — this
        is how a rebind after a device replug happens without a restart.
        """
        with self._lock:
            count = len(self._loaded)
            self._loaded.clear()
            return count

    # -- discovery ---------------------------------------------------------

    def _labels_path(self, name: str) -> Path | None:
        # User-supplied per-model files first, then the guessed shared zoo label
        # file, then a generic shared labels.txt.
        candidates = [
            self.model_dir / f"{name}.labels.txt",
            self.model_dir / f"{name}.txt",
        ]
        guess = zoo_label_file(name)
        if guess:
            candidates.append(self.model_dir / guess)
        candidates.append(self.model_dir / "labels.txt")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def list(self) -> list[ModelInfo]:
        if not self.model_dir.is_dir():
            return []
        return [
            ModelInfo(
                name=p.stem,
                # Filename only: the absolute server path is not disclosed.
                path=p.name,
                size_bytes=p.stat().st_size,
                edgetpu_compiled=p.stem.endswith("_edgetpu"),
                has_labels=self._labels_path(p.stem) is not None,
                loaded=p.stem in self._loaded,
            )
            for p in sorted(self.model_dir.glob("*.tflite"))
        ]

    # -- acquisition ---------------------------------------------------------

    def _fetch(self, url: str, dest: Path, max_bytes: int) -> bool:
        """Stream url to dest atomically, capped at max_bytes. False on 404,
        raises on other errors.

        Redirects are not followed: a redirect to another host would let the
        model source steer a server-side request anywhere (SSRF), and a
        redirect on the same host has no legitimate use here. The byte cap (the
        declared Content-Length, then a running total as a fallback) stops a
        huge download from filling the models volume.
        """
        tmp = dest.with_name(dest.name + ".part")
        try:
            with httpx.stream("GET", url, follow_redirects=False, timeout=120) as resp:
                if resp.status_code == 404:
                    return False
                if resp.is_redirect:
                    raise ModelDownloadError(
                        f"Refusing redirect while downloading {url} "
                        f"(-> {resp.headers.get('location')!r})"
                    )
                resp.raise_for_status()
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise ModelDownloadError(
                        f"{url} declares {declared} bytes, over the {max_bytes}-byte limit"
                    )
                written = 0
                with tmp.open("wb") as f:
                    for chunk in resp.iter_bytes():
                        written += len(chunk)
                        if written > max_bytes:
                            raise ModelDownloadError(
                                f"{url} exceeds the {max_bytes}-byte download limit"
                            )
                        f.write(chunk)
            tmp.replace(dest)
            return True
        except httpx.HTTPError as exc:
            raise ModelDownloadError(f"Failed to download {url}: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)

    def _downloaded_count(self) -> int:
        if not self.model_dir.is_dir():
            return 0
        return sum(1 for _ in self.model_dir.glob("*.tflite"))

    def _cache_bytes(self) -> int:
        if not self.model_dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.model_dir.glob("*.tflite"))

    def _verify_checksum(self, name: str, path: Path) -> None:
        """If a sha256 is pinned for this model, verify it; delete + raise on
        mismatch. No-op when no checksum is configured."""
        expected = self.checksums.get(name)
        if not expected:
            return
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            path.unlink(missing_ok=True)
            raise ModelDownloadError(f"Checksum mismatch for model {name!r}")

    def _ensure_local(self, name: str) -> Path:
        path = self.model_dir / f"{name}.tflite"
        if path.is_file():
            return path
        if not self.model_download:
            raise ModelNotFound(name)
        if self._downloaded_count() >= self.max_downloaded_models:
            raise ModelDownloadError(
                f"Model cache is full ({self.max_downloaded_models} models); "
                "refusing to download more. Remove unused models from model_dir."
            )
        if self._cache_bytes() >= self.max_model_cache_bytes:
            raise ModelDownloadError(
                f"Model cache is full ({self.max_model_cache_bytes} bytes); "
                "refusing to download more. Remove unused models from model_dir."
            )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if not self._fetch(f"{self.model_source}/{name}.tflite", path, self.max_model_bytes):
            raise ModelNotFound(name)
        self._verify_checksum(name, path)
        # Best-effort auto-fetch of the model's shared label file (coco_labels,
        # imagenet_labels, …) so `label` is populated without manual setup. The
        # file is shared across models, so it's downloaded once and reused.
        label_file = zoo_label_file(name)
        if label_file and not (self.model_dir / label_file).is_file():
            try:
                self._fetch(
                    f"{self.model_source}/{label_file}",
                    self.model_dir / label_file,
                    _LABEL_MAX_BYTES,
                )
            except ModelDownloadError:
                pass
        return path

    def get(self, name: str) -> LoadedModel:
        """Return a loaded model, downloading and loading it if needed.

        Blocking (network + TPU); call it from the TPU executor, not the
        event loop.
        """
        if not _NAME_RE.fullmatch(name):
            raise ModelNotFound(name)
        with self._lock:
            if name in self._loaded:
                return self._loaded[name]
            path = self._ensure_local(name)
            labels_path = self._labels_path(name)
            model = LoadedModel(
                name=name,
                path=path,
                device=self.device,
                edgetpu_lib=self.edgetpu_lib,
                labels=load_labels(labels_path) if labels_path else None,
            )
            self._loaded[name] = model
            return model
