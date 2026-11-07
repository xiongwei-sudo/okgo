"""进程内任务队列：一卡一任务（concurrency=1）。"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class JobState:
    job_id: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    message: str = ""
    output: Optional[Path] = None
    error: Optional[str] = None


class GPUJobQueue:
    """单 worker 线程池，保证同一时刻仅一个 GPU job。"""

    def __init__(self, max_workers: int = 1):
        if max_workers != 1:
            logger.warning("强制 concurrency=1/GPU，忽略 max_workers=%s", max_workers)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-job")
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[JobState], None], job_id: Optional[str] = None) -> str:
        jid = job_id or uuid.uuid4().hex[:12]
        state = JobState(job_id=jid)
        with self._lock:
            self._jobs[jid] = state

        def _wrap() -> None:
            state.status = JobStatus.running
            try:
                fn(state)
                if state.status != JobStatus.failed:
                    state.status = JobStatus.done
                    state.progress = 1.0
            except Exception as e:
                logger.exception("job %s failed", jid)
                state.status = JobStatus.failed
                state.error = str(e)

        self._pool.submit(_wrap)
        return jid

    def get(self, job_id: str) -> Optional[JobState]:
        return self._jobs.get(job_id)
