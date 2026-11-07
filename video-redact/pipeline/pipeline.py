"""主流水线：可选增强 →（可选）检测 → 跟踪 → 打码 → 编码；ffmpeg 回嵌音频。

enhance-only：未勾选 face/plate/text 时跳过检测与打码，只写出增强结果。
"""
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
from .queue import JobCancelled
from .track import SimpleTracker

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[float, str], None]]
ControlCB = Optional[Callable[[], None]]


@dataclass
class RedactOptions:
    face: bool = True
    plate: bool = True
    text: bool = True
    mode: str = "gaussian"  # gaussian|mosaic|solid
    strength: float = 0.75
    margin: float = 0.12
    stride: int = 1
    device: str = "cuda:0"
    keep_audio: bool = True
    unload_after: bool = True
    enhance: EnhanceOptions = field(default_factory=EnhanceOptions)


@dataclass
class JobResult:
    output: Path
    frames: int
    detections_total: int
    notes: List[str] = field(default_factory=list)
    enhance_paths: dict = field(default_factory=dict)


def _mux_audio(src: Path, silent_video: Path, dst: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg 未安装，跳过音频回嵌")
        shutil.copy2(silent_video, dst)
        return False
    cmd = [
        ffmpeg, "-y",
        "-i", str(silent_video),
        "-i", str(src),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
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

    kinds = []
    if opts.face:
        kinds.append("face")
    if opts.plate:
        kinds.append("plate")
    if opts.text:
        kinds.append("text")
    do_redact = bool(kinds)
    do_enhance = bool(opts.enhance.enable)

    if not do_redact and not do_enhance:
        raise ValueError("至少启用 face/plate/text 之一，或启用增强")

    # 仅增强时可写出真实超分分辨率；有打码时保持同分辨率以免框坐标错位
    if do_enhance and not do_redact:
        opts.enhance.keep_resolution = False
    elif do_enhance and do_redact:
        opts.enhance.keep_resolution = True

    _cp()
    enhance = EnhancePipeline(weights_dir, device=opts.device)
    if do_enhance:
        if progress:
            progress(0.02, "加载增强模型")
        _cp()
        enhance.load_for(opts.enhance)
        notes.extend(opts.enhance.notes)
        if progress:
            summary = ", ".join(f"{k}={v}" for k, v in opts.enhance.path_report.items()) or "none"
            progress(0.04, f"enhance_path: {summary}")
        _cp()
    else:
        notes.append("enhance_path: skipped")

    hub = None
    detectors = []
    if do_redact:
        if progress:
            progress(0.03, "加载检测模型")
        _cp()
        hub = DetectorHub(weights_dir, device=opts.device)
        detectors = hub.get(kinds)
        _cp()
        if opts.plate and getattr(hub.plate, "_backend", None) == "heuristic":
            notes.append("车牌权重【待补】，已用启发式降级")
    else:
        notes.append("打码已关闭（enhance-only），跳过人脸/车牌/OCR 检测与马赛克")
        logger.info("enhance-only: skip face/plate/text redact")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 先读一帧确定输出尺寸（超分可能放大）
    _cp()
    ok0, frame0 = cap.read()
    if not ok0:
        cap.release()
        raise RuntimeError("视频无帧")
    if do_enhance:
        frame0 = enhance.process_frame(frame0, opts.enhance)
    out_h, out_w = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_dir = Path(tempfile.mkdtemp(prefix="redact_"))
    silent_path = tmp_dir / "silent.mp4"
    writer = cv2.VideoWriter(str(silent_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败")

    tracker = SimpleTracker() if do_redact else None
    det_count = 0
    frame_idx = 0
    last_boxes = []
    cancelled = False

    pbar = tqdm(total=total if total > 0 else None, desc="enhance" if not do_redact else "redact", unit="f")
    try:
        try:
            while True:
                if control is not None:
                    control()
                if frame_idx == 0:
                    frame = frame0
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if do_enhance:
                        frame = enhance.process_frame(frame, opts.enhance)

                if do_redact:
                    assert tracker is not None
                    # 若增强改了分辨率，检测在增强后分辨率上进行
                    if opts.stride <= 1 or frame_idx % opts.stride == 0:
                        boxes = []
                        for det in detectors:
                            for box, label, score in det.detect(frame):
                                boxes.append(box)
                                det_count += 1
                        tracked = tracker.update(boxes)
                        last_boxes = tracked
                    else:
                        last_boxes = tracker.update([])
                    out = redact_frame(
                        frame,
                        [(int(a), int(b), int(c), int(d)) for a, b, c, d in last_boxes],
                        mode=opts.mode,  # type: ignore[arg-type]
                        strength=opts.strength,
                        margin=opts.margin,
                    )
                else:
                    out = frame

                if out.shape[1] != out_w or out.shape[0] != out_h:
                    out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_AREA)
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
            if hub is not None:
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
        output=output_path,
        frames=frame_idx,
        detections_total=det_count,
        notes=notes,
        enhance_paths=dict(opts.enhance.path_report),
    )
