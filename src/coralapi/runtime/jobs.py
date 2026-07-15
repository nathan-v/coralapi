"""In-memory store for asynchronous inference jobs.

Jobs live in process memory: results survive until job_retention_seconds after
completion, then are purged lazily. Run replicas stateless — a job must be
polled on the replica that accepted it (use session affinity if scaling out).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from coralapi import metrics
from coralapi.errors import public_error_message


@dataclass
class Job:
    kind: str
    model: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "queued"  # queued | running | succeeded | failed
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: Any = None
    error: str | None = None


class JobStore:
    def __init__(self, retention_seconds: float = 3600):
        self.retention_seconds = retention_seconds
        self._jobs: dict[str, Job] = {}

    def _purge(self) -> None:
        cutoff = time.time() - self.retention_seconds
        for job_id in [
            j.id
            for j in self._jobs.values()
            if j.finished_at is not None and j.finished_at < cutoff
        ]:
            del self._jobs[job_id]

    def create(self, kind: str, model: str) -> Job:
        self._purge()
        job = Job(kind=kind, model=model)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        self._purge()
        return self._jobs.get(job_id)

    def start(
        self,
        job: Job,
        run_reserved: Callable[[Callable[[], Any]], Awaitable[Any]],
        fn: Callable[[], Any],
        cleanup: Callable[[], None] | None = None,
    ) -> asyncio.Task:
        """Schedule a job on the (already reserved) TPU executor slot."""

        def marked_running() -> Any:
            job.status = "running"
            return fn()

        async def runner() -> None:
            started = time.time()
            try:
                job.result = await run_reserved(marked_running)
                job.status = "succeeded"
                metrics.record_inference(job.kind, job.model, "ok", time.time() - started)
            except Exception as exc:  # job outcome is reported via polling
                job.status = "failed"
                job.error = public_error_message(exc)
                metrics.record_inference(job.kind, job.model, "error", None)
            finally:
                job.finished_at = time.time()
                if cleanup is not None:
                    cleanup()

        return asyncio.create_task(runner())
