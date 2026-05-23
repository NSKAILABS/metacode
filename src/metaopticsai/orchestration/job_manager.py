"""Background job manager — 

Threading is sufficient for the typical workload (one heavy job at a time on
a developer laptop / single-tenant server). For multi-tenant or distributed
deployments, swap for Celery or RQ behind the same JobManager API.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
import traceback
import uuid
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Job:
    """A single submitted job."""
    id: str
    kind: str
    status: str = "queued"        # queued | running | sampling | training | done | failed
    submitted_at: float = dataclasses.field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class JobManager:
    """Thread-safe job registry. One JobManager per process is typical."""

    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}

    def submit(
        self,
        kind: str,
        target: Callable[..., Any],
        *args: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a background job. Returns job_id immediately."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, kind=kind, metadata=metadata or {})
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._runner,
            args=(job_id, target, args, kwargs),
            daemon=True,
            name=f"job-{kind}-{job_id}",
        )
        thread.start()
        return job_id

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"No job {job_id!r}")
        return job

    def list(self, status: str | None = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def update_status(self, job_id: str, status: str) -> None:
        """Used by long jobs to publish progress like 'sampling' → 'training'."""
        with self._lock:
            j = self._jobs.get(job_id)
            if j is not None:
                j.status = status

    # ── runner ─────────────────────────────────────────────────────────────

    def _runner(
        self,
        job_id: str,
        target: Callable[..., Any],
        args: tuple,
        kwargs: dict,
    ) -> None:
        # Pass the manager into the target as a kwarg so the worker can
        # call `manager.update_status(job_id, "training")`.
        kwargs.setdefault("_manager", self)
        kwargs.setdefault("_job_id", job_id)

        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
        try:
            result = target(*args, **kwargs)
            with self._lock:
                job.status = "done"
                job.result = result
                job.finished_at = time.time()
        except Exception as e:  # noqa: BLE001
            log.exception("Job %s failed", job_id)
            with self._lock:
                job.status = "failed"
                job.error = f"{e}\n{traceback.format_exc()}"
                job.finished_at = time.time()