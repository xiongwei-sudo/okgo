"""主流水线：可选增强 stub → 检测 → 跟踪 → 打码 → 编码；ffmpeg 回嵌音频。"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
from tqdm import tqdm

from .blur import redact_frame
from .detectors import DetectorHub
from .enhance import EnhanceOptions, EnhancePipeline
from .queue import JobCancelled
from .track import SimpleTracker

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[float, str], None]]
ControlCB = Optional[Callable[[], None]]  # 帧间：暂停阻塞 / 取消抛 JobCancelled


@dataclass
class RedactOptions:
    face: bool = True
    plate: bool = True
    text: bool = True
    mode: str = "gaussian"  # gaussian|mosaic|solid
    strength: float = 0.75
    margin: float = 0.12
    stride: int = 1  # 检测步长；中间帧靠跟踪
    device: str = "cuda:0"
    keep_audio: bool = True
    unload_after: bool = True  # 按需加载 / 任务结束卸载
    enhance: EnhanceOptions = field(default_factory=EnhanceOptions)


@dataclass
class JobResult:
    output: Path
    frames: int
    detections_total: int
    notes: List[str] = field(default_factory=list)


def _mux_audio(src: Path, silent_video: Path, dst: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg 未安装，跳过音频回嵌")
        shutil.copy2(silent_video, dst)
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.warning("ffmpeg mux failed: %s", e.stderr[-500:] if e.stderr else e)
        shutil.copy2(silent_video, dst)
        return False


def run_redact(
    input_path: Path,
    output_path: Path,
    weights_dir: Path,
    opts: RedactOptions,
    progress: ProgressCB = None,
    control: ControlCB = None,
) -> JobResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []

    def _cp() -> None:
        if control is not None:
            control()

    _cp()
    enhance = EnhancePipeline(weights_dir, device=opts.device)
    if opts.enhance.enable:
        if progress:
            progress(0.02, "加载增强模型")
        _cp()
        enhance.load_for(opts.enhance)
        notes.extend(opts.enhance.notes)
        _cp()

    kinds = []
    if opts.face:
        kinds.append("face")
    if opts.plate:
        kinds.append("plate")
    if opts.text:
        kinds.append("text")
    if not kinds:
        raise ValueError("至少启用 face/plate/text 之一")

    if progress:
        progress(0.03, "加载检测模型")
    _cp()
    hub = DetectorHub(weights_dir, device=opts.device)
    detectors = hub.get(kinds)
    _cp()
    if opts.plate and hub.plate._backend == "heuristic":
        notes.append("车牌权重【待补】，已用启发式降级")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_dir = Path(tempfile.mkdtemp(prefix="redact_"))
    silent_path = tmp_dir / "silent.mp4"
    writer = cv2.VideoWriter(str(silent_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败")

    tracker = SimpleTracker()
    det_count = 0
    frame_idx = 0
    last_boxes = []

    pbar = tqdm(total=total if total > 0 else None, desc="redact", unit="f")
    cancelled = False
    try:
        try:
            while True:
                if control is not None:
                    control()  # 帧间安全点：暂停阻塞 / 取消抛出
                ok, frame = cap.read()
                if not ok:
                    break
                frame = enhance.process_frame(frame, opts.enhance)

                if opts.stride <= 1 or frame_idx % opts.stride == 0:
                    boxes = []
                    for det in detectors:
                        for box, label, score in det.detect(frame):
                            boxes.append(box)
                            det_count += 1
                    tracked = tracker.update(boxes)
                    last_boxes = tracked
                else:
                    # 非检测帧：沿用跟踪预测（简易：上一框）
                    last_boxes = tracker.update([])

                out = redact_frame(
                    frame,
                    [(int(a), int(b), int(c), int(d)) for a, b, c, d in last_boxes],
                    mode=opts.mode,  # type: ignore[arg-type]
                    strength=opts.strength,
                    margin=opts.margin,
                )
                writer.write(out)
                frame_idx += 1
                pbar.update(1)
                if progress and total > 0:
                    progress(frame_idx / total, f"frame {frame_idx}/{total}")
                if control is not None:
                    control()
        except JobCancelled:
            cancelled = True
            notes.append("任务已取消")
            raise
    finally:
        pbar.close()
        cap.release()
        writer.release()
        if opts.unload_after:
            hub.unload_all()
            enhance.unload()
        if cancelled:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if opts.keep_audio:
        muxed = _mux_audio(input_path, silent_path, output_path)
        if not muxed:
            notes.append("无音频或 ffmpeg 不可用，已输出无音轨视频")
    else:
        shutil.copy2(silent_path, output_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return JobResult(
        output=output_path, frames=frame_idx, detections_total=det_count, notes=notes
    )
