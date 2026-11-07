"""进程内任务队列：一卡一任务（concurrency=1），支持取消 / 暂停 / 继续。"""
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
    paused = "paused"
    cancelled = "cancelled"
    done = "done"
    failed = "failed"


class JobCancelled(Exception):
    """协作式取消：流水线在帧边界抛出。"""


@dataclass
class JobState:
    job_id: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    message: str = ""
    output: Optional[Path] = None
    error: Optional[str] = None
    # 控制信号：cancel 置位即停；pause_gate 为 Event，set=可跑，clear=暂停
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    pause_gate: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        # 默认允许运行
        if not self.pause_gate.is_set():
            self.pause_gate.set()


class GPUJobQueue:
    """单 worker 线程池，保证同一时刻仅一个 GPU job。"""

    def __init__(self, max_workers: int = 1):
        if max_workers != 1:
            logger.warning("强制 concurrency=1/GPU，忽略 max_workers=%s", max_workers)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-job")
        self._jobs: Dict[str, JobState] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[JobState], None], job_id: Optional[str] = None) -> str:
        jid = job_id or uuid.uuid4().hex[:12]
        state = JobState(job_id=jid)
        with self._lock:
            self._jobs[jid] = state

        def _wrap() -> None:
            # 若在排队阶段已被取消，直接退出
            if state.cancel_flag.is_set() or state.status == JobStatus.cancelled:
                state.status = JobStatus.cancelled
                state.message = "已取消"
                return
            # 排队暂停：在真正开跑前阻塞（帧间安全点的排队等价物）
            try:
                self.wait_if_paused(state)
            except JobCancelled:
                state.status = JobStatus.cancelled
                state.message = "已取消"
                return
            if state.cancel_flag.is_set() or state.status == JobStatus.cancelled:
                state.status = JobStatus.cancelled
                state.message = "已取消"
                return
            state.status = JobStatus.running
            state.message = state.message or "处理中"
            try:
                fn(state)
                if state.cancel_flag.is_set() or state.status == JobStatus.cancelled:
                    state.status = JobStatus.cancelled
                    state.message = "已取消"
                elif state.status not in (JobStatus.failed, JobStatus.cancelled):
                    state.status = JobStatus.done
                    state.progress = 1.0
            except JobCancelled:
                state.status = JobStatus.cancelled
                state.message = "已取消"
                logger.info("job %s cancelled", jid)
            except Exception as e:
                if state.cancel_flag.is_set():
                    state.status = JobStatus.cancelled
                    state.message = "已取消"
                    state.error = str(e)
                else:
                    logger.exception("job %s failed", jid)
                    state.status = JobStatus.failed
                    state.error = str(e)

        fut = self._pool.submit(_wrap)
        with self._lock:
            self._futures[jid] = fut
        return jid

    def get(self, job_id: str) -> Optional[JobState]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> tuple[bool, str]:
        """取消排队或运行中的任务。运行中在下一帧安全点停止并卸载模型。"""
        with self._lock:
            state = self._jobs.get(job_id)
            fut = self._futures.get(job_id)
        if not state:
            return False, "not found"
        if state.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
            return False, f"cannot cancel status={state.status.value}"

        state.cancel_flag.set()
        # 若处于暂停，放行以便立刻感知取消
        state.pause_gate.set()
        state.message = "正在取消…"

        if state.status == JobStatus.queued and fut is not None:
            cancelled = fut.cancel()
            if cancelled:
                state.status = JobStatus.cancelled
                state.message = "已取消（未开始）"
                return True, "cancelled_queued"

        # 运行中 / 暂停：协作式停止，由 worker 在帧边界抛 JobCancelled
        if state.status == JobStatus.paused:
            state.status = JobStatus.running  # 让循环走起来感知 cancel
        return True, "cancel_requested"

    def pause(self, job_id: str) -> tuple[bool, str]:
        """暂停：在下一帧安全点停下；排队中的任务暂不启动（清 pause_gate）。"""
        state = self._jobs.get(job_id)
        if not state:
            return False, "not found"
        if state.status == JobStatus.queued:
            state.pause_gate.clear()
            state.status = JobStatus.paused
            state.message = "已暂停（排队中，下一安全点生效）"
            return True, "paused_queued"
        if state.status == JobStatus.running:
            state.pause_gate.clear()
            state.status = JobStatus.paused
            state.message = "暂停请求已发，将在下一帧安全点停下"
            return True, "pause_requested"
        return False, f"cannot pause status={state.status.value}"

    def resume(self, job_id: str) -> tuple[bool, str]:
        state = self._jobs.get(job_id)
        if not state:
            return False, "not found"
        if state.status != JobStatus.paused and not (
            state.status == JobStatus.queued and not state.pause_gate.is_set()
        ):
            # 也允许 status 仍是 running 但 gate 被清的边缘情况
            if state.pause_gate.is_set() and state.status != JobStatus.paused:
                return False, f"cannot resume status={state.status.value}"
        if state.cancel_flag.is_set():
            return False, "job cancelled"
        state.pause_gate.set()
        if state.status == JobStatus.paused:
            # 排队暂停 → 回到 queued；运行暂停 → running
            fut = self._futures.get(job_id)
            started = fut is not None and fut.running()
            state.status = JobStatus.running if started else JobStatus.queued
            state.message = "已继续" if started else "已继续排队"
        return True, "resumed"

    def wait_if_paused(self, state: JobState) -> None:
        """流水线帧间调用：暂停则阻塞；取消则抛 JobCancelled。"""
        if state.cancel_flag.is_set():
            state.status = JobStatus.cancelled
            state.message = "已取消"
            raise JobCancelled("cancelled")
        # 暂停时阻塞，直到 resume 或 cancel
        while not state.pause_gate.wait(timeout=0.25):
            if state.cancel_flag.is_set():
                state.pause_gate.set()
                state.status = JobStatus.cancelled
                state.message = "已取消"
                raise JobCancelled("cancelled")
            # 保持 paused 状态可见
            if state.status != JobStatus.paused:
                state.status = JobStatus.paused
                state.message = state.message or "已暂停（帧间安全点）"

        if state.cancel_flag.is_set():
            state.status = JobStatus.cancelled
            state.message = "已取消"
            raise JobCancelled("cancelled")

        # 从暂停恢复
        if state.status == JobStatus.paused:
            state.status = JobStatus.running
            state.message = "继续处理"
