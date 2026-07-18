# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly. For issues that do not require private disclosure, please open a GitHub issue.

For sensitive security issues that require private disclosure, please [contact](https://www.nathanv.com/contact) the maintainer directly.

Please include:
- A description of the vulnerability
- Steps to reproduce the issue
- Any potential impact
- Suggested fix (if you have one)

You should receive a response within 72 hours. Security fixes will be prioritized and released as soon as possible.

## Known Limitations

CoralAPI ships **without authentication or rate limiting by design** — it is meant to run behind an authenticated gateway or ingress that provides those. See the [Security model](README.md#security-model) section of the README for the full posture. In summary:

- **No auth / no rate limiting**: the trust boundary is a fronting gateway. Do not expose the service directly to untrusted networks.
- **Async job IDs are bearer secrets**: job results are readable by anyone holding the (unguessable, UUID4) job ID; there is no per-caller ownership.
- **Model downloads** run server-side against a configured HTTPS source with redirects disabled, per-model and total-cache byte caps, and optional sha256 pinning (`CORALAPI_MODEL_CHECKSUMS`). Point `CORALAPI_MODEL_SOURCE` only at a source you trust; models are loaded onto the TPU.
- **Uploads** are streamed to disk with a hard size limit enforced at the ASGI layer and a maximum image resolution; tune `CORALAPI_MAX_UPLOAD_BYTES` and `CORALAPI_MAX_IMAGE_DIM` for your environment.
