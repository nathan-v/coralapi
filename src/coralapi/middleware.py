"""ASGI middleware enforcing a hard request-body ceiling.

FastAPI/Starlette stream multipart file parts into an unbounded spooled temp
file *before* the endpoint runs, so an in-endpoint size check happens too late
to protect the host. This middleware caps the body at the transport layer:

  - a declared Content-Length over the limit is rejected before any body is
    read (413), and
  - a chunked/streaming body is aborted mid-transfer once the running total
    exceeds the limit, so a client that omits or lies about Content-Length
    still cannot fill the disk.

It is deliberately independent of auth/rate-limiting: it is the one backstop
that bounds unauthenticated resource use on the upload path.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_413 = 413
_413_BODY = b'{"detail":"Request body exceeds the configured size limit"}'


async def _reject_413(send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": _413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_413_BODY)).encode()),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": _413_BODY})


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: reject an over-limit declared Content-Length outright.
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await _reject_413(send)
                        return
                except ValueError:
                    pass
                break

        received = 0
        exceeded = False
        swallowing = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    # Stop feeding the body to the app and signal end-of-stream;
                    # the app sees a truncated body and its own read stops.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message: Message) -> None:
            # If we tripped the limit mid-body, replace the app's entire
            # response with a single 413 and swallow whatever it emits.
            nonlocal swallowing
            if swallowing:
                return
            if exceeded and message["type"] == "http.response.start":
                swallowing = True
                await _reject_413(send)
                return
            await send(message)

        await self.app(scope, limited_receive, guarded_send)
