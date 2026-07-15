"""Streamed upload handling.

Uploads are copied to a temp file in bounded chunks so a large body never
lives in memory, and are rejected mid-stream once they exceed the configured
size limit. Image dimensions are validated from the header before any pixel
data is decoded.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

CHUNK_BYTES = 1024 * 1024


async def save_upload(file: UploadFile, max_bytes: int, upload_dir: Path | None) -> Path:
    """Stream an upload to disk; 413 if it exceeds max_bytes."""
    fd = tempfile.NamedTemporaryFile(prefix="coralapi-upload-", dir=upload_dir, delete=False)
    path = Path(fd.name)
    size = 0
    try:
        with fd:
            while chunk := await file.read(CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Upload exceeds the {max_bytes} byte limit")
                fd.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def open_image_checked(path: Path, max_dim: int) -> Image.Image:
    """Open an image lazily and validate its dimensions from the header.

    Pixel decoding is deferred until the inference worker touches the data,
    so this is cheap to call on the request path. PIL's decompression-bomb
    ceiling is aligned to the configured limit instead of PIL's default.
    """
    Image.MAX_IMAGE_PIXELS = max_dim * max_dim
    try:
        image = Image.open(path)
    except UnidentifiedImageError:
        raise HTTPException(400, "Uploaded file is not a decodable image") from None
    except Image.DecompressionBombError:
        # PIL pre-empts our check for images far beyond the pixel ceiling.
        raise HTTPException(
            413, f"Image resolution exceeds the limit; the longest edge may not exceed {max_dim}px"
        ) from None
    width, height = image.size
    if width > max_dim or height > max_dim:
        raise HTTPException(
            413,
            f"Image is {width}x{height}; the longest edge may not exceed {max_dim}px",
        )
    return image
