"""Background job manager.

Purpose
-------
Runs long admin tasks off the request thread and lets the UI track their progress.

What it does
------------
Submits tasks to a thread pool, assigns each a job record, and exposes status,
listing, and active-job queries. Sized for a local or small-LAN deployment.

Flow
----
``submit()`` queues a task and returns a job handle; the worker updates the job's
status and result as it runs; the UI polls ``get()``/``list()`` until the job
completes.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from backend.app.jobs.models import JobRecord, utc_now_iso


class JobManager:
    """Thread-backed job registry for local admin operations."""

    def __init__(self, max_workers: int = 1, keep_last: int = 50):
        self.max_workers = max(1, int(max_workers))
        self.keep_last = max(10, int(keep_last))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="rags-admin-job")
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}

    def resize(self, max_workers: int) -> None:
        """Resize future job capacity by replacing the executor if needed."""

        max_workers = max(1, int(max_workers))
        with self._lock:
            if max_workers == self.max_workers:
                return
            old = self._executor
            self.max_workers = max_workers
            self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rags-admin-job")
            old.shutdown(wait=False, cancel_futures=False)

    def submit(self, name: str, func: Callable[[], Any]) -> JobRecord:
        """Submit a job and return the queued JobRecord immediately."""

        with self._lock:
            self._prune_locked()
            job_id = f"job-{uuid4().hex[:12]}"
            record = JobRecord(job_id=job_id, name=name)
            self._jobs[job_id] = record

            future = self._executor.submit(self._run_job, job_id, func)
            self._futures[job_id] = future
            return record

    def _run_job(self, job_id: str, func: Callable[[], Any]) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = utc_now_iso()

        try:
            result = func()
            with self._lock:
                record = self._jobs[job_id]
                record.status = "completed"
                record.result = result
                record.finished_at = utc_now_iso()
        except Exception as exc:
            with self._lock:
                record = self._jobs[job_id]
                record.status = "failed"
                record.error = str(exc)
                record.finished_at = utc_now_iso()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[JobRecord]:
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return records[: max(1, int(limit))]

    def active_jobs(self) -> list[JobRecord]:
        with self._lock:
            return [job for job in self._jobs.values() if job.status in {"queued", "running"}]

    def has_active_job(self, names: set[str] | None = None) -> bool:
        active = self.active_jobs()
        if names is None:
            return bool(active)
        return any(job.name in names for job in active)


    def shutdown(self) -> None:
        """Shutdown the executor during FastAPI app shutdown."""

        with self._lock:
            self._executor.shutdown(wait=False, cancel_futures=False)

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self.keep_last:
            return

        records = sorted(self._jobs.values(), key=lambda item: item.created_at)
        removable = [job for job in records if job.status in {"completed", "failed"}]
        for job in removable[: max(0, len(self._jobs) - self.keep_last)]:
            self._jobs.pop(job.job_id, None)
            self._futures.pop(job.job_id, None)
