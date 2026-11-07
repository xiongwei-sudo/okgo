"""增强模块占位：N1 Real-ESRGAN / N2 低光·去噪 / N3 稳像 / N4 BasicVSR++。

默认链路建议：稳像 → 低光 → 去噪/去糊 → 超分，再进入打码。
一期 CLI 仅提供透传 stub，权重与推理【待补】。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
    scale: int = 2  # 2 or 4
    notes: List[str] = field(default_factory=list)


class EnhancePipeline:
    """按需加载增强模型；当前为 stub，原样返回帧并记录待补项。"""

    def __init__(self, weights_dir: Path, device: str = "cuda:0"):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self._loaded: List[str] = []

    def load_for(self, opts: EnhanceOptions) -> None:
        if not opts.enable:
            return
        pending = []
        if opts.stabilize:
            pending.append("N3-vidstab/OpenCV【待补】")
        if opts.lowlight:
            pending.append("N2-Zero-DCE++【待补】")
        if opts.denoise:
            pending.append("N2-NAFNet【待补】")
        if opts.deblur:
            pending.append("N2-NAFNet-Deblur【待补】")
        if opts.superres:
            pending.append(f"N1-Real-ESRGAN-x{opts.scale}【待补】")
        opts.notes.extend(pending)
        for p in pending:
            logger.warning("enhance stub: %s", p)
        self._loaded = pending

    def unload(self) -> None:
        self._loaded = []

    def process_frame(self, frame: np.ndarray, opts: EnhanceOptions) -> np.ndarray:
        if not opts.enable:
            return frame
        # stub: identity
        return frame

    def process_video_path(
        self, input_path: Path, output_path: Path, opts: EnhanceOptions
    ) -> Optional[Path]:
        """整段增强入口；未实现时返回 None，由上层跳过。"""
        if not opts.enable:
            return None
        self.load_for(opts)
        logger.info("EnhancePipeline stub skip full-video rewrite; identity path")
        return None
