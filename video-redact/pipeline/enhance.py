"""增强模块：N1 Real-ESRGAN / N2 低光·去噪 / N3 稳像 / N4 BasicVSR++。

默认链路建议：稳像 → 低光 → 去噪/去糊 → 超分，再进入打码。
专用权重缺失时走 OpenCV 降级，并打日志【待补】。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhanceOptions:
    enable: bool = False
    stabilize: bool = False  # N3 vidstab / OpenCV
    lowlight: bool = False  # N2 Zero-DCE++
    denoise: bool = False  # N2 NAFNet
    deblur: bool = False  # N2 NAFNet-Deblur
    superres: bool = False  # N1 Real-ESRGAN
    scale: int = 2  # 2 or 4（真超分需权重；OpenCV 降级默认同分辨率增强）
    notes: List[str] = field(default_factory=list)


def _has_weight(weights_dir: Path, patterns: Tuple[str, ...]) -> Optional[Path]:
    root = Path(weights_dir)
    for pat in patterns:
        hits = list(root.glob(pat)) + list((root / "enhance").glob(pat))
        for h in hits:
            if h.is_file() and h.stat().st_size > 1000:
                return h
    return None


class EnhancePipeline:
    """按需加载增强模型；缺权重时 OpenCV 降级。"""

    def __init__(self, weights_dir: Path, device: str = "cuda:0"):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self._loaded: List[str] = []
        self._warned: set[str] = set()
        # 稳像状态（逐帧光流）
        self._prev_gray: Optional[np.ndarray] = None
        self._transform_acc = np.eye(2, 3, dtype=np.float64)

    def load_for(self, opts: EnhanceOptions) -> None:
        if not opts.enable:
            return
        pending: List[str] = []
        if opts.stabilize:
            if _has_weight(self.weights_dir, ("*vidstab*", "*stabil*", "*.npz")):
                pending.append("N3-vidstab 权重已发现但仍【待补】接入")
            else:
                pending.append("N3-vidstab【待补】→ OpenCV 光流稳像降级")
        if opts.lowlight:
            if _has_weight(self.weights_dir, ("*Zero-DCE*", "*zero_dce*", "*dce*")):
                pending.append("N2-Zero-DCE++ 权重已发现但仍【待补】接入")
            else:
                pending.append("N2-Zero-DCE++【待补】→ OpenCV CLAHE 降级")
        if opts.denoise:
            if _has_weight(self.weights_dir, ("*NAFNet*", "*nafnet*", "*denoise*")):
                pending.append("N2-NAFNet 权重已发现但仍【待补】接入")
            else:
                pending.append("N2-NAFNet【待补】→ OpenCV 去噪降级")
        if opts.deblur:
            if _has_weight(self.weights_dir, ("*Deblur*", "*deblur*")):
                pending.append("N2-NAFNet-Deblur 权重已发现但仍【待补】接入")
            else:
                pending.append("N2-NAFNet-Deblur【待补】→ OpenCV 锐化降级")
        if opts.superres:
            if _has_weight(
                self.weights_dir,
                ("*RealESRGAN*", "*realesrgan*", "*ESRGAN*", f"*x{opts.scale}*"),
            ):
                pending.append(f"N1-Real-ESRGAN-x{opts.scale} 权重已发现但仍【待补】接入")
            else:
                pending.append(
                    f"N1-Real-ESRGAN-x{opts.scale}【待补】→ OpenCV 同分辨率增强降级"
                )
        # 去重写入 notes
        for p in pending:
            if p not in opts.notes:
                opts.notes.append(p)
            if p not in self._warned:
                logger.warning("%s", p)
                self._warned.add(p)
        self._loaded = pending

    def unload(self) -> None:
        self._loaded = []
        self._prev_gray = None
        self._transform_acc = np.eye(2, 3, dtype=np.float64)
        self._warned.clear()

    def _stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if self._prev_gray is None:
            self._prev_gray = gray
            return frame
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )
        if prev_pts is None or len(prev_pts) < 8:
            self._prev_gray = gray
            return frame
        nxt_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
        if nxt_pts is None or status is None:
            self._prev_gray = gray
            return frame
        ok_prev = prev_pts[status.flatten() == 1]
        ok_next = nxt_pts[status.flatten() == 1]
        if len(ok_prev) < 8:
            self._prev_gray = gray
            return frame
        m, _inliers = cv2.estimateAffinePartial2D(ok_prev, ok_next)
        if m is None:
            self._prev_gray = gray
            return frame
        # 反向补偿：把当前帧拉回上一参考
        try:
            inv = cv2.invertAffineTransform(m)
        except cv2.error:
            self._prev_gray = gray
            return frame
        # 轻微平滑累积，避免漂移过大
        self._transform_acc[:, :2] = (
            0.9 * self._transform_acc[:, :2] + 0.1 * inv[:, :2]
        )
        self._transform_acc[:, 2] = 0.9 * self._transform_acc[:, 2] + 0.1 * inv[:, 2]
        stabilized = cv2.warpAffine(
            frame,
            self._transform_acc,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        self._prev_gray = gray
        return stabilized

    def _lowlight(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # 大图用更快的 bilateral，小图可用 NL-means
        if h * w > 1280 * 720:
            return cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)
        return cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)

    def _deblur(self, frame: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.2)
        return cv2.addWeighted(frame, 1.6, blur, -0.6, 0)

    def _superres_same_size(self, frame: np.ndarray, scale: int) -> np.ndarray:
        """真超分【待补】；降级：放大再缩回 + 锐化，不改变输出分辨率（兼容现有 writer）。"""
        h, w = frame.shape[:2]
        s = max(2, int(scale))
        up = cv2.resize(frame, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
        sharp = self._deblur(up)
        return cv2.resize(sharp, (w, h), interpolation=cv2.INTER_AREA)

    def process_frame(self, frame: np.ndarray, opts: EnhanceOptions) -> np.ndarray:
        if not opts.enable:
            return frame
        out = frame
        # 顺序：稳像 → 低光 → 去噪 → 去糊 → 超分降级
        if opts.stabilize:
            out = self._stabilize_frame(out)
        if opts.lowlight:
            out = self._lowlight(out)
        if opts.denoise:
            out = self._denoise(out)
        if opts.deblur:
            out = self._deblur(out)
        if opts.superres:
            out = self._superres_same_size(out, opts.scale)
        return out

    def process_video_path(
        self, input_path: Path, output_path: Path, opts: EnhanceOptions
    ) -> Optional[Path]:
        """整段增强入口；用 OpenCV 写出增强视频。真超分放大尺寸时写入新分辨率。"""
        if not opts.enable:
            return None
        self.load_for(opts)
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error("EnhancePipeline 无法打开: %s", input_path)
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 真超分权重【待补】时仍同分辨率；若将来接入 Real-ESRGAN 再乘 scale
        out_w, out_h = width, height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            logger.error("EnhancePipeline VideoWriter 失败")
            return None
        # 重置稳像状态
        self._prev_gray = None
        self._transform_acc = np.eye(2, 3, dtype=np.float64)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out = self.process_frame(frame, opts)
                if out.shape[1] != out_w or out.shape[0] != out_h:
                    out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(out)
        finally:
            cap.release()
            writer.release()
        logger.info("EnhancePipeline OpenCV 降级写出: %s", output_path)
        return output_path
