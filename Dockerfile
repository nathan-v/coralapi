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
# The downloads are checksum-verified (supply chain): when you override
# LIBEDGETPU_DEB or a URL TFLITE_SPEC, also override the matching *_SHA256
# (compute with `sha256sum <file>`). A PyPI TFLITE_SPEC (name==ver) is resolved
# and integrity-checked by uv against the index, so TFLITE_SHA256 is unused.
ARG LIBEDGETPU_DEB=https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/libedgetpu1-std_16.0tf2.17.1-1.bookworm_amd64.deb
ARG LIBEDGETPU_SHA256=a3e1f9ae3e4725032a863ea38f52aa396ee18fdb65135fed56e8efae5c2e467d
ARG TFLITE_SPEC=tflite-runtime @ https://github.com/feranick/TFlite-builds/releases/download/v2.17.1/tflite_runtime-2.17.1-cp311-cp311-linux_x86_64.whl
ARG TFLITE_SHA256=5983bc00bc47d86a6a39f258ad07b69bd4d9fd61703535b10347c13f6901eb05

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL "$LIBEDGETPU_DEB" -o /tmp/libedgetpu.deb \
    && echo "${LIBEDGETPU_SHA256}  /tmp/libedgetpu.deb" | sha256sum -c - \
    && apt-get install -y --no-install-recommends /tmp/libedgetpu.deb \
    && rm /tmp/libedgetpu.deb \
    && rm -rf /var/lib/apt/lists/*

# uv is pinned (not :latest) so the builder tool is reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Install the selected tflite_runtime, plus the project with the 'prometheus'
# extra (so /metrics can be enabled via env with no rebuild) but WITHOUT the
# 'tpu' extra (which would pull the incompatible PyPI tflite-runtime). A URL
# wheel is downloaded and checksum-verified before install; a PyPI spec goes
# straight to uv. Then curl is removed (ca-certificates stays for runtime
# model downloads over https).
RUN set -eux; \
    case "${TFLITE_SPEC}" in \
      *://*) url="${TFLITE_SPEC##* }"; \
             curl -fsSL "$url" -o /tmp/tflite.whl; \
             echo "${TFLITE_SHA256}  /tmp/tflite.whl" | sha256sum -c -; \
             uv pip install --system --no-cache /tmp/tflite.whl ".[prometheus]"; \
             rm /tmp/tflite.whl ;; \
      *) uv pip install --system --no-cache "${TFLITE_SPEC}" ".[prometheus]" ;; \
    esac; \
    apt-get purge -y curl && apt-get autoremove -y

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
