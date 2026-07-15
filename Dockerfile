# coralapi — inference API for Google Coral Edge TPUs.
#
# The image must run with the TPU passed through:
#   PCIe/M.2:  docker run --device /dev/apex_0 ...
#   USB:       docker run --device /dev/bus/usb ...
# Without a TPU the service exits at startup by design.

# Python 3.11: matches the cp311 tflite_runtime wheel installed below.
FROM python:3.11-slim-bookworm

# ---- Runtime profile (build-time selectable) --------------------------------
# libedgetpu and tflite_runtime MUST be built against compatible TFLite
# versions, or the Edge TPU delegate loads but fails at invoke
# ("EdgeTpuDelegateForCustomOp failed to invoke"). Also, prebuilt tflite wheels
# target a CPU instruction set — an AVX2 wheel SIGILLs on a pre-Haswell CPU.
#
# Select a profile for your target by overriding these two ARGs (see the README
# "Build profiles" table). Defaults = AVX2 x86_64 (fastest; needs Haswell+).
#
#   avx2 (default): feranick libedgetpu 2.17.1 + feranick tflite_runtime 2.17.1.
#   compat (no-AVX2 x86, e.g. Sandy/Ivy Bridge Xeon) — build with:
#     --build-arg LIBEDGETPU_DEB=https://github.com/feranick/libedgetpu/releases/download/v16.0TF2.13.1-2/libedgetpu1-std_16.0tf2.13.1-2.bookworm_amd64.deb
#     --build-arg TFLITE_SPEC=tflite-runtime==2.14.0
#   arm (Raspberry Pi) — also pass --platform linux/arm/v7, and build with:
#     --build-arg LIBEDGETPU_DEB=https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/libedgetpu1-std_16.0tf2.17.1-1.bookworm_armhf.deb
#     --build-arg TFLITE_SPEC=tflite-runtime @ https://github.com/feranick/TFlite-builds/releases/download/v2.17.1/tflite_runtime-2.17.1-cp311-cp311-linux_armv7l.whl
#
# std = standard clock; feranick also ships *-max .debs for ~2x speed at higher
# power/heat. TFLITE_SPEC is any pip requirement: a "name @ URL" or "name==ver".
ARG LIBEDGETPU_DEB=https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/libedgetpu1-std_16.0tf2.17.1-1.bookworm_amd64.deb
ARG TFLITE_SPEC=tflite-runtime @ https://github.com/feranick/TFlite-builds/releases/download/v2.17.1/tflite_runtime-2.17.1-cp311-cp311-linux_x86_64.whl

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL "$LIBEDGETPU_DEB" -o /tmp/libedgetpu.deb \
    && apt-get install -y --no-install-recommends /tmp/libedgetpu.deb \
    && rm /tmp/libedgetpu.deb \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Install the selected tflite_runtime, plus the project with the 'prometheus'
# extra (so /metrics can be enabled via env with no rebuild) but WITHOUT the
# 'tpu' extra (which would pull the incompatible PyPI tflite-runtime).
RUN uv pip install --system --no-cache "${TFLITE_SPEC}" ".[prometheus]"

# Run as an unprivileged user. The TPU device node is typically owned by a
# host group (e.g. 'apex' for PCIe, 'plugdev' for USB) — pass that group at
# runtime with `--group-add <gid>` (Docker) or an fsGroup/supplementalGroups
# securityContext (Kubernetes) so the non-root user can open the device.
RUN useradd --system --create-home --uid 10001 coral \
    && mkdir -p /models \
    && chown coral:coral /models
VOLUME /models

USER coral

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "coralapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
