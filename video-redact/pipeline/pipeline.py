"""主流水线：可选增强（无官方权重时 OpenCV 降级）→ 检测 → 跟踪 → 打码 → 编码；ffmpeg 回嵌音频。"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
from tqdm import tqdm

from .blur import redact_frame
from .detectors import DetectorHub
from .enhance import EnhanceOptions, EnhancePipeline
from .track import SimpleTracker

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[float, str], None]]


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
) -> JobResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []

    enhance = EnhancePipeline(weights_dir, device=opts.device)
    tmp_dir = Path(tempfile.mkdtemp(prefix="redact_"))
    source = input_path
    cap = None
    writer = None
    pbar = None
    hub = None
    frame_idx = 0
    det_count = 0
    silent_path = tmp_dir / "silent.mp4"
    try:
        try:
            if opts.enhance.enable:
                enhance.load_for(opts.enhance)
                source = enhance.consume_stabilize_prepass(input_path, tmp_dir, opts.enhance)
                notes.extend(opts.enhance.notes)

            kinds = []
            if opts.face:
                kinds.append("face")
            if opts.plate:
                kinds.append("plate")
            if opts.text:
                kinds.append("text")
            if not kinds:
                raise ValueError("至少启用 face/plate/text 之一")

            hub = DetectorHub(weights_dir, device=opts.device)
            detectors = hub.get(kinds)
            if opts.plate and hub.plate._backend == "heuristic":
                notes.append("车牌权重【待补】，已用启发式降级（见 weights/PLATE_WEIGHTS_PENDING.txt）")

            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {source}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            scale = enhance.output_scale(opts.enhance)
            out_w, out_h = width * scale, height * scale

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(silent_path), fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter 打开失败")

            tracker = SimpleTracker()
            last_boxes: list = []
            pbar = tqdm(total=total if total > 0 else None, desc="redact", unit="f")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = enhance.process_frame(frame, opts.enhance)
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

                if opts.stride <= 1 or frame_idx % opts.stride == 0:
                    boxes = []
                    for det in detectors:
                        for box, _label, _score in det.detect(frame):
                            boxes.append(box)
                            det_count += 1
                    last_boxes = tracker.update(boxes)
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
                if progress:
                    if total > 0:
                        progress(frame_idx / total, f"frame {frame_idx}/{total}")
                    else:
                        progress(0.0, f"frame {frame_idx}")
        finally:
            if pbar is not None:
                pbar.close()
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if opts.unload_after:
                if hub is not None:
                    hub.unload_all()
                enhance.unload()

        if opts.keep_audio:
            muxed = _mux_audio(input_path, silent_path, output_path)
            if not muxed:
                notes.append("无音频或 ffmpeg 不可用，已输出无音轨视频")
        else:
            shutil.copy2(silent_path, output_path)
        return JobResult(
            output=output_path, frames=frame_idx, detections_total=det_count, notes=notes
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
