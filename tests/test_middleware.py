"""ASGI-level tests for BodySizeLimitMiddleware, exercised without a server."""

import asyncio

from coralapi.middleware import BodySizeLimitMiddleware


async def ok_app(scope, receive, send):
    """Minimal app that drains the body and returns 200 with its byte count."""
    total = 0
    while True:
        msg = await receive()
        total += len(msg.get("body", b""))
        if not msg.get("more_body", False):
            break
    body = str(total).encode()
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": body})


def run(app, scope, chunks):
    """Drive an ASGI app with a list of body chunks; capture sent messages."""
    sent = []
    it = iter(chunks)

    async def receive():
        try:
            chunk = next(it)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def http_scope(headers=None):
    return {"type": "http", "headers": headers or []}


def test_declared_content_length_over_limit_rejected_before_body():
    body_read = False

    async def tracking_app(scope, receive, send):
        nonlocal body_read
        await receive()
        body_read = True

    app = BodySizeLimitMiddleware(tracking_app, max_bytes=10)
    sent = run(app, http_scope([(b"content-length", b"1000")]), [b"x" * 1000])

    assert sent[0]["status"] == 413
    assert body_read is False  # rejected without ever reading the body


def test_chunked_body_over_limit_aborted_midstream():
    # No Content-Length header: the limit must be enforced from the stream.
    app = BodySizeLimitMiddleware(ok_app, max_bytes=10)
    sent = run(app, http_scope(), [b"x" * 6, b"x" * 6, b"x" * 6])

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    # The app's real 200 response must not leak through after the 413.
    assert all(m.get("status", 413) == 413 for m in sent if m["type"] == "http.response.start")


def test_body_within_limit_passes_through():
    app = BodySizeLimitMiddleware(ok_app, max_bytes=100)
    sent = run(app, http_scope([(b"content-length", b"6")]), [b"abcdef"])

    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b"6"


def test_non_http_scope_passes_through():
    seen = {}

    async def lifespan_app(scope, receive, send):
        seen["type"] = scope["type"]

    app = BodySizeLimitMiddleware(lifespan_app, max_bytes=10)
    asyncio.run(app({"type": "lifespan"}, None, None))
    assert seen["type"] == "lifespan"
