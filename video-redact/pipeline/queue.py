"""进程内任务队列：默认同卡串行，可选有限并行（显存门禁），支持取消 / 暂停 / 继续。

调度判断机制（scheduling guards）：
- 提交前根据 worker 存活、心跳、占用与 GPU 空闲决定 run / queue / reject
- pause / cancel / restart / reupload 后 reconcile，避免「空闲却卡住 queued」
- 任务与 worker 状态以真实线程/心跳为准，不只看 job flag
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    paused = "paused"
    cancelled = "cancelled"
    done = "done"
    failed = "failed"


class WorkerLiveness(str, Enum):
    idle = "idle"  # 池活着且无在跑任务
    busy = "busy"  # 有活跃任务且心跳正常
    hung = "hung"  # 标称运行中但心跳超时
    dead = "dead"  # 线程池不可用 / 需重建


class AdmitDecision(str, Enum):
    run = "run"  # 立刻占槽开跑（响应层仍先 queued，worker 马上接手）
    queue = "queue"  # 槽位占用，进入排队
    reject = "reject"  # 无法启动（worker 挂死 / GPU 不可用等）


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
    # 进程/调度元数据
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    heartbeat_at: Optional[float] = None
    worker_tid: Optional[int] = None
    worker_ident: Optional[int] = None
    admit: Optional[str] = None
    admit_reason: str = ""
    wants_enhance: bool = False
    fail_reason: Optional[str] = None
    enhance_paths: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pause_gate.is_set():
            self.pause_gate.set()

    def touch(self) -> None:
        self.heartbeat_at = time.time()


def _probe_gpu() -> Dict[str, Any]:
    """探测 GPU 是否可用 / 是否有计算进程。失败不抛，返回可读原因。"""
    info: Dict[str, Any] = {
        "available": False,
        "idle": True,
        "reason": "",
        "compute_pids": [],
        "memory_used_mb": None,
        "memory_total_mb": None,
        "name": None,
    }
    # 1) nvidia-smi
    smi = None
    for cand in ("nvidia-smi", "/usr/bin/nvidia-smi"):
        if cand == "nvidia-smi" or Path(cand).exists():
            smi = cand if cand != "nvidia-smi" else "nvidia-smi"
            break
    if smi:
        try:
            out = subprocess.check_output(
                [
                    smi,
                    "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
                timeout=5,
                text=True,
            ).strip()
            if out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                if len(parts) >= 3:
                    info["name"] = parts[0]
                    try:
                        info["memory_used_mb"] = float(parts[1])
                        info["memory_total_mb"] = float(parts[2])
                    except ValueError:
                        pass
                    info["available"] = True
            apps = subprocess.check_output(
                [
                    smi,
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
                timeout=5,
                text=True,
            ).strip()
            pids: List[int] = []
            if apps:
                for line in apps.splitlines():
                    cols = [c.strip() for c in line.split(",")]
                    if cols and cols[0].isdigit():
                        pids.append(int(cols[0]))
            # 过滤本进程自身（worker 占用不算「别人占卡」时仍标 busy）
            self_pid = os.getpid()
            foreign = [p for p in pids if p != self_pid]
            info["compute_pids"] = pids
            info["idle"] = len(foreign) == 0
            if foreign:
                info["reason"] = f"GPU 被其他进程占用: pids={foreign}"
            elif not info["available"]:
                info["reason"] = "nvidia-smi 无 GPU 行"
            return info
        except FileNotFoundError:
            pass
        except Exception as e:
            info["reason"] = f"nvidia-smi 失败: {e}"
    # 2) torch fallback
    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            try:
                free, total = torch.cuda.mem_get_info(0)
                info["memory_used_mb"] = round((total - free) / (1024 * 1024), 1)
                info["memory_total_mb"] = round(total / (1024 * 1024), 1)
                # 无法区分外进程；有明显占用则不算完全 idle
                info["idle"] = (total - free) < 256 * 1024 * 1024
                if not info["idle"]:
                    info["reason"] = "CUDA 显存已有占用（可能为本进程或其他）"
            except Exception:
                info["idle"] = True
            return info
        info["reason"] = info["reason"] or "torch.cuda 不可用"
    except Exception as e:
        info["reason"] = info["reason"] or f"无 GPU 探测: {e}"
    if not info["reason"]:
        info["reason"] = "未检测到 NVIDIA GPU"
    return info


class GPUJobQueue:
    """单 worker 线程池，保证同一时刻仅一个 GPU job；带心跳与调度判断。"""

    def __init__(
        self,
        max_workers: int = 1,
        hung_seconds: float = 90.0,
        require_gpu_for_enhance: bool = False,
        min_free_vram_mb_per_job: float = 6000.0,
    ):
        # 默认同卡串行；可通过 max_workers>1 开启有限并行（显存门禁）
        self._max_workers = max(1, min(int(max_workers), 2))  # 硬顶 2，防 OOM
        if self._max_workers > 1:
            logger.warning(
                "GPU concurrency=%s（显存门禁 %.0fMB/任务）；请监控 nvidia-smi",
                self._max_workers,
                min_free_vram_mb_per_job,
            )
        self._hung_seconds = float(hung_seconds)
        self._require_gpu_for_enhance = bool(require_gpu_for_enhance)
        self._min_free_vram_mb_per_job = float(min_free_vram_mb_per_job)
        self._pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="gpu-job")
        self._pool_gen = 0
        self._jobs: Dict[str, JobState] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.RLock()
        self._current_job_id: Optional[str] = None
        self._worker_heartbeat: float = time.time()
        self._worker_tid: Optional[int] = None
        self._worker_alive_flag = True
        self._server_pid = os.getpid()
        # 后台轻量 reconcile，避免「槽空却 queued」
        self._stop_watchdog = threading.Event()
        # 对账周期不超过 hung 阈值的 1/3，保证卡死能被及时发现
        self._watchdog_interval = max(0.5, min(2.0, self._hung_seconds / 3.0))
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="gpu-queue-watchdog", daemon=True
        )
        self._watchdog.start()

    # ------------------------------------------------------------------ utils
    def _active_statuses(self) -> Tuple[JobStatus, ...]:
        return (JobStatus.queued, JobStatus.running, JobStatus.paused)

    def _count_active(self) -> int:
        return sum(1 for s in self._jobs.values() if s.status in self._active_statuses())

    def _running_or_paused(self) -> List[JobState]:
        return [
            s
            for s in self._jobs.values()
            if s.status in (JobStatus.running, JobStatus.paused)
        ]

    def _queued_waiting(self) -> List[JobState]:
        # 排队且 pause_gate 开着（未被「排队暂停」挡住）的任务
        out = []
        for s in self._jobs.values():
            if s.status == JobStatus.queued and s.pause_gate.is_set() and not s.cancel_flag.is_set():
                out.append(s)
        out.sort(key=lambda x: x.created_at)
        return out

    def _future_alive(self, job_id: str) -> bool:
        fut = self._futures.get(job_id)
        if fut is None:
            return False
        return not fut.done()

    def _mark_failed(self, state: JobState, reason: str) -> None:
        state.status = JobStatus.failed
        state.fail_reason = reason
        state.error = reason
        state.message = reason
        state.finished_at = time.time()
        if self._current_job_id == state.job_id:
            self._current_job_id = None

    def heartbeat(self, state: Optional[JobState] = None) -> None:
        now = time.time()
        self._worker_heartbeat = now
        self._worker_alive_flag = True
        tid = threading.get_ident()
        self._worker_tid = tid
        if state is not None:
            state.touch()
            state.worker_ident = tid
            try:
                state.worker_tid = threading.get_native_id()  # type: ignore[attr-defined]
            except Exception:
                state.worker_tid = tid

    # --------------------------------------------------------------- liveness
    def liveness(self) -> WorkerLiveness:
        """根据真实心跳 / 占槽任务判定 worker 状态。"""
        with self._lock:
            self._reconcile_unlocked()
            active_exec = self._running_or_paused()
            now = time.time()
            if not self._worker_alive_flag and active_exec:
                return WorkerLiveness.dead
            for st in active_exec:
                hb = st.heartbeat_at or st.started_at or st.created_at
                # 暂停时心跳会在 wait 循环里刷新；若完全失联则 hung
                if (now - hb) > self._hung_seconds and self._future_alive(st.job_id):
                    return WorkerLiveness.hung
                if not self._future_alive(st.job_id) and st.status in (
                    JobStatus.running,
                    JobStatus.paused,
                ):
                    return WorkerLiveness.dead
            if active_exec:
                return WorkerLiveness.busy
            return WorkerLiveness.idle

    def worker_snapshot(self) -> Dict[str, Any]:
        gpu = _probe_gpu()
        with self._lock:
            self._reconcile_unlocked()
            live = self.liveness()
            cur = self._current_job_id
            cur_st = self._jobs.get(cur) if cur else None
            queued = self._queued_waiting()
            active = [s for s in self._jobs.values() if s.status in self._active_statuses()]
            return {
                "liveness": live.value,
                "alive": live in (WorkerLiveness.idle, WorkerLiveness.busy),
                "hung": live == WorkerLiveness.hung,
                "dead": live == WorkerLiveness.dead,
                "server_pid": self._server_pid,
                "worker_tid": self._worker_tid,
                "worker_heartbeat_age_s": round(time.time() - self._worker_heartbeat, 2),
                "hung_threshold_s": self._hung_seconds,
                "concurrency": self._max_workers,
                "slot_free": len(self._running_or_paused()) < self._max_workers,
                "current_job_id": cur,
                "current_status": cur_st.status.value if cur_st else None,
                "active_jobs": len(active),
                "queued_jobs": len(queued),
                "queued_job_ids": [s.job_id for s in queued],
                "pool_generation": self._pool_gen,
                "gpu": gpu,
            }

    # --------------------------------------------------------- admit / decide
    def admit(
        self, *, wants_enhance: bool = False, force_queue: bool = False
    ) -> Tuple[AdmitDecision, str]:
        """提交前判断：run / queue / reject，并给出可读原因。"""
        with self._lock:
            self._reconcile_unlocked()
            live = self.liveness()
            gpu = _probe_gpu()

            if live == WorkerLiveness.dead:
                # 尝试自愈重建池
                self._rebuild_pool_unlocked("admit: worker dead")
                live = self.liveness()
                if live == WorkerLiveness.dead:
                    return (
                        AdmitDecision.reject,
                        "worker 进程/线程已死亡，无法启动任务；请重启服务或稍后重试",
                    )

            if live == WorkerLiveness.hung:
                return (
                    AdmitDecision.reject,
                    f"worker 心跳超时（>{self._hung_seconds:.1f}s）疑似卡死；"
                    "请先取消/重启服务释放 GPU，再提交",
                )

            if wants_enhance:
                if self._require_gpu_for_enhance and not gpu.get("available"):
                    return (
                        AdmitDecision.reject,
                        f"增强任务需要 GPU，但当前不可用：{gpu.get('reason') or 'unknown'}",
                    )
                # GPU 被外部进程占满且本队列空闲 → 增强无法真正开跑
                if (
                    live == WorkerLiveness.idle
                    and gpu.get("available")
                    and not gpu.get("idle")
                    and gpu.get("compute_pids")
                ):
                    foreign = [p for p in gpu["compute_pids"] if p != self._server_pid]
                    if foreign:
                        return (
                            AdmitDecision.reject,
                            f"GPU 空闲槽被其他进程占用（pids={foreign}），增强无法启动",
                        )

            if force_queue:
                return AdmitDecision.queue, "强制排队"

            running = self._running_or_paused()
            if running:
                if len(running) < self._max_workers:
                    # 并行槽：检查剩余显存
                    free_mb = None
                    try:
                        used = gpu.get("memory_used_mb")
                        total = gpu.get("memory_total_mb")
                        if used is not None and total is not None:
                            free_mb = float(total) - float(used)
                    except Exception:
                        free_mb = None
                    need = self._min_free_vram_mb_per_job
                    if free_mb is not None and free_mb < need:
                        holders = [f"{s.job_id}:{s.status.value}" for s in running]
                        return (
                            AdmitDecision.queue,
                            f"并行槽可用但显存不足（free≈{free_mb:.0f}MB < {need:.0f}MB），排队（{', '.join(holders)}）",
                        )
                    # 允许第二任务 run
                    return (
                        AdmitDecision.run,
                        f"并行开跑（{len(running)+1}/{self._max_workers}），剩余显存≈{free_mb if free_mb is not None else '?'}MB",
                    )
                holders = [
                    f"{s.job_id}:{s.status.value}" for s in running
                ]
                return (
                    AdmitDecision.queue,
                    f"GPU worker 忙碌（{', '.join(holders) or 'busy'}），任务将排队",
                )

            # idle 且有尚未被线程池取走的 queued —— 仍应 queue（保持 FIFO）
            waiting = self._queued_waiting()
            if waiting:
                return (
                    AdmitDecision.queue,
                    f"前方仍有 {len(waiting)} 个排队任务，按序等待",
                )

            return AdmitDecision.run, "worker 空闲且存活，立即开跑"

    # ------------------------------------------------------------ reconcile
    def _reconcile_unlocked(self) -> List[str]:
        """对照 Future/心跳修正状态；必要时重建池，避免 queued 永久卡住。"""
        notes: List[str] = []
        now = time.time()

        for jid, st in list(self._jobs.items()):
            fut = self._futures.get(jid)
            if st.status in (JobStatus.running, JobStatus.paused):
                hb = st.heartbeat_at or st.started_at or st.created_at
                if fut is not None and fut.done():
                    # future 结束但状态未收尾
                    if st.status not in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
                        exc = None
                        try:
                            exc = fut.exception()
                        except Exception as e:
                            exc = e
                        if st.cancel_flag.is_set():
                            st.status = JobStatus.cancelled
                            st.message = st.message or "已取消"
                            notes.append(f"{jid}: future done → cancelled")
                        elif exc:
                            self._mark_failed(st, f"worker 异常退出: {exc}")
                            notes.append(f"{jid}: future done → failed")
                        else:
                            # 正常结束但未置 done
                            st.status = JobStatus.done
                            st.progress = 1.0
                            st.message = st.message or "完成"
                            st.finished_at = now
                            notes.append(f"{jid}: future done → done")
                        if self._current_job_id == jid:
                            self._current_job_id = None
                elif (now - hb) > self._hung_seconds:
                    # 心跳超时：标 hung 失败并准备换池
                    reason = (
                        f"worker 无心跳超过 {self._hung_seconds:.1f}s，判定卡死并释放槽位"
                    )
                    self._mark_failed(st, reason)
                    notes.append(f"{jid}: hung → failed")
                    self._rebuild_pool_unlocked(f"hung job {jid}")
            elif st.status == JobStatus.queued:
                # 已取消标记但还停在 queued
                if st.cancel_flag.is_set():
                    st.status = JobStatus.cancelled
                    st.message = st.message or "已取消"
                    st.finished_at = now
                    notes.append(f"{jid}: queued+cancel → cancelled")
                elif fut is not None and fut.done() and fut.cancelled():
                    st.status = JobStatus.cancelled
                    st.message = st.message or "已取消（未开始）"
                    st.finished_at = now
                    notes.append(f"{jid}: future cancelled")

        # 槽位空闲时，确认仍有 queued future 未被丢掉（ThreadPool 会自然调度；
        # 若 future 丢失则重新提交包装是危险的，这里只记录诊断）
        if not self._running_or_paused():
            waiting = self._queued_waiting()
            for st in waiting:
                fut = self._futures.get(st.job_id)
                if fut is None:
                    self._mark_failed(
                        st, "排队任务丢失 worker future（槽位空闲却无法启动）"
                    )
                    notes.append(f"{st.job_id}: queued missing future → failed")
                elif fut.done() and st.status == JobStatus.queued:
                    self._mark_failed(
                        st, "排队任务 future 已结束但从未开跑"
                    )
                    notes.append(f"{st.job_id}: queued future done oddly → failed")

        if notes:
            logger.info("reconcile: %s", "; ".join(notes))
        return notes

    def reconcile(self) -> Dict[str, Any]:
        with self._lock:
            notes = self._reconcile_unlocked()
            snap = self.worker_snapshot()
            snap["reconcile_notes"] = notes
            return snap

    def _rebuild_pool_unlocked(self, why: str) -> None:
        logger.warning("rebuild ThreadPoolExecutor: %s", why)
        try:
            self._pool.shutdown(wait=False, cancel_futures=False)
        except TypeError:
            # py3.8 compat
            self._pool.shutdown(wait=False)
        except Exception:
            logger.exception("pool shutdown failed")
        self._pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="gpu-job")
        self._pool_gen += 1
        self._current_job_id = None
        self._worker_tid = None
        self._worker_heartbeat = time.time()
        self._worker_alive_flag = True
        # 旧 running 已在调用方标 failed；把仍 queued 且 future 死掉的任务重新投递
        for st in list(self._jobs.values()):
            if st.status != JobStatus.queued or st.cancel_flag.is_set():
                continue
            old = self._futures.get(st.job_id)
            if old is not None and not old.done():
                # 旧池上的 future 不会在新池执行 → 必须失败提示用户重启/重提
                self._mark_failed(
                    st,
                    f"worker 池已重建（{why}），原排队任务失效，请重新提交",
                )

    def _watchdog_loop(self) -> None:
        while not self._stop_watchdog.wait(self._watchdog_interval):
            try:
                with self._lock:
                    self._reconcile_unlocked()
            except Exception:
                logger.exception("watchdog reconcile failed")

    # ---------------------------------------------------------------- submit
    def submit(
        self,
        fn: Callable[[JobState], None],
        job_id: Optional[str] = None,
        *,
        wants_enhance: bool = False,
        admit: Optional[AdmitDecision] = None,
        admit_reason: str = "",
    ) -> str:
        jid = job_id or uuid.uuid4().hex[:12]
        state = JobState(job_id=jid, wants_enhance=wants_enhance)
        if admit is not None:
            state.admit = admit.value
            state.admit_reason = admit_reason
            state.message = admit_reason or (
                "排队中" if admit == AdmitDecision.queue else "等待 GPU worker"
            )
        with self._lock:
            self._reconcile_unlocked()
            self._jobs[jid] = state

        def _wrap() -> None:
            self.heartbeat(state)
            # 若在排队阶段已被取消，直接退出
            if state.cancel_flag.is_set() or state.status == JobStatus.cancelled:
                state.status = JobStatus.cancelled
                state.message = state.message or "已取消"
                state.finished_at = time.time()
                return
            # 排队暂停：在真正开跑前阻塞
            try:
                self.wait_if_paused(state)
            except JobCancelled:
                state.status = JobStatus.cancelled
                state.message = state.message or "已取消"
                state.finished_at = time.time()
                return
            if state.cancel_flag.is_set() or state.status == JobStatus.cancelled:
                state.status = JobStatus.cancelled
                state.message = state.message or "已取消"
                state.finished_at = time.time()
                return

            with self._lock:
                self._current_job_id = jid
            state.status = JobStatus.running
            state.started_at = time.time()
            state.message = state.message or "处理中"
            self.heartbeat(state)
            try:
                fn(state)
                self.heartbeat(state)
                if state.cancel_flag.is_set() or state.status == JobStatus.cancelled:
                    state.status = JobStatus.cancelled
                    state.message = state.message or "已取消"
                elif state.status not in (JobStatus.failed, JobStatus.cancelled):
                    state.status = JobStatus.done
                    state.progress = 1.0
                    if not state.message:
                        state.message = "完成"
            except JobCancelled:
                state.status = JobStatus.cancelled
                state.message = state.message or "已取消"
                logger.info("job %s cancelled", jid)
            except Exception as e:
                if state.cancel_flag.is_set():
                    state.status = JobStatus.cancelled
                    state.message = state.message or "已取消"
                    state.error = str(e)
                else:
                    logger.exception("job %s failed", jid)
                    state.status = JobStatus.failed
                    state.error = str(e)
                    state.fail_reason = str(e)
                    state.message = str(e)
            finally:
                state.finished_at = time.time()
                with self._lock:
                    if self._current_job_id == jid:
                        self._current_job_id = None
                    # 任务结束后立刻 reconcile，让下一 queued 不被错误状态挡住
                    self._reconcile_unlocked()

        fut = self._pool.submit(_wrap)
        with self._lock:
            self._futures[jid] = fut
        return jid

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            self._reconcile_unlocked()
            return self._jobs.get(job_id)

    # ----------------------------------------------------------- controls
    def cancel(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
            state = self._jobs.get(job_id)
            fut = self._futures.get(job_id)
            if not state:
                return False, "not found"
            if state.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
                return False, f"cannot cancel status={state.status.value}"

            state.cancel_flag.set()
            state.pause_gate.set()
            state.message = "正在取消…"

            if state.status == JobStatus.queued and fut is not None:
                cancelled = fut.cancel()
                if cancelled:
                    state.status = JobStatus.cancelled
                    state.message = "已取消（未开始）"
                    state.finished_at = time.time()
                    self._reconcile_unlocked()
                    return True, "cancelled_queued"

            if state.status == JobStatus.paused:
                state.status = JobStatus.running
            self._reconcile_unlocked()
            return True, "cancel_requested"

    def pause(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
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
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return False, "not found"
            if state.status != JobStatus.paused and not (
                state.status == JobStatus.queued and not state.pause_gate.is_set()
            ):
                if state.pause_gate.is_set() and state.status != JobStatus.paused:
                    return False, f"cannot resume status={state.status.value}"
            if state.cancel_flag.is_set():
                return False, "job cancelled"
            state.pause_gate.set()
            if state.status == JobStatus.paused:
                fut = self._futures.get(job_id)
                started = fut is not None and fut.running()
                # 真实进程态：future running → running，否则回 queued 等槽
                if started:
                    state.status = JobStatus.running
                    state.message = "已继续"
                else:
                    state.status = JobStatus.queued
                    state.message = "已继续排队（等待 GPU 空闲槽）"
                self._reconcile_unlocked()
            return True, "resumed"

    def wait_if_paused(self, state: JobState) -> None:
        """流水线帧间调用：暂停则阻塞；取消则抛 JobCancelled；并刷新心跳。"""
        self.heartbeat(state)
        if state.cancel_flag.is_set():
            state.status = JobStatus.cancelled
            state.message = state.message or "已取消"
            raise JobCancelled("cancelled")
        while not state.pause_gate.wait(timeout=0.25):
            self.heartbeat(state)
            if state.cancel_flag.is_set():
                state.pause_gate.set()
                state.status = JobStatus.cancelled
                state.message = state.message or "已取消"
                raise JobCancelled("cancelled")
            if state.status != JobStatus.paused:
                state.status = JobStatus.paused
                state.message = state.message or "已暂停（帧间安全点）"

        self.heartbeat(state)
        if state.cancel_flag.is_set():
            state.status = JobStatus.cancelled
            state.message = state.message or "已取消"
            raise JobCancelled("cancelled")

        if state.status == JobStatus.paused:
            state.status = JobStatus.running
            state.message = "继续处理"
