"""Bounded serial access to the Edge TPU, pinned to one dedicated thread.

All TPU work — delegate creation, interpreter construction, and invocation —
must run on a single, consistent OS thread: libedgetpu delegates are
thread-affine, so a delegate created on one thread cannot be invoked from
another (it fails at invoke with "EdgeTpuDelegateForCustomOp failed to
invoke"). A max_workers=1 ThreadPoolExecutor gives us that single thread and
serializes work for free (the TPU runs one inference at a time anyway).

A counter bounds how many requests may wait for the thread; beyond that,
submission fails fast with QueueFull so callers get backpressure (HTTP 429)
instead of unbounded latency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class QueueFull(RuntimeError):
    pass


class TPUExecutor:
    def __init__(self, queue_depth: int):
        self.queue_depth = queue_depth
        self._capacity = queue_depth + 1  # one running + queue_depth waiting
        self._pending = 0
        # A single worker thread owns the TPU. Every task below runs on it.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coral-tpu")

    @property
    def pending(self) -> int:
        """Requests currently running or waiting for the TPU."""
        return self._pending

    def reserve(self) -> None:
        """Claim a queue slot synchronously; raises QueueFull if none is free.

        Callers must follow up with run_reserved(), which releases the slot.
        """
        if self._pending >= self._capacity:
            raise QueueFull(f"TPU queue is full ({self.queue_depth} waiting); retry later")
        self._pending += 1

    async def run_reserved(self, fn: Callable[[], Any]) -> Any:
        """Run fn on the dedicated TPU thread once free; releases the slot."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._pool, fn)
        finally:
            self._pending -= 1

    async def submit(self, fn: Callable[[], Any]) -> Any:
        """reserve() + run_reserved() for synchronous request paths."""
        self.reserve()
        return await self.run_reserved(fn)

    def run_sync(self, fn: Callable[[], Any]) -> Any:
        """Run fn on the TPU thread and block for the result.

        For startup-time work (the Edge TPU probe) that must happen on the
        same thread all later inference runs on, before the event loop serves
        requests.
        """
        return self._pool.submit(fn).result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
