"""打码渲染：高斯模糊 / 马赛克 / 纯色遮罩。"""
from __future__ import annotations

from typing import Iterable, Literal, Sequence, Tuple

import cv2
import numpy as np

Mode = Literal["gaussian", "mosaic", "solid"]
BBox = Tuple[int, int, int, int]  # x1,y1,x2,y2


def _clip_box(box: BBox, w: int, h: int) -> BBox | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(box: BBox, margin: float, w: int, h: int) -> BBox | None:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * margin, bh * margin
    return _clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), w, h)


def apply_blur_region(
    frame: np.ndarray,
    box: BBox,
    mode: Mode = "gaussian",
    strength: float = 0.7,
) -> None:
    """原地打码一个区域。strength 0~1。"""
    h, w = frame.shape[:2]
    clipped = _clip_box(box, w, h)
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    strength = float(np.clip(strength, 0.05, 1.0))
    if mode == "gaussian":
        k = max(3, int(min(x2 - x1, y2 - y1) * (0.15 + 0.55 * strength)))
        if k % 2 == 0:
            k += 1
        k = min(k, 99)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    elif mode == "mosaic":
        block = max(4, int(8 + 40 * (1.0 - strength)))
        small = cv2.resize(roi, (max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)))
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    else:  # solid
        frame[y1:y2, x1:x2] = (0, 0, 0)


def redact_frame(
    frame: np.ndarray,
    boxes: Sequence[BBox],
    mode: Mode = "gaussian",
    strength: float = 0.7,
    margin: float = 0.12,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for box in boxes:
        expanded = expand_box(tuple(box), margin, w, h)
        if expanded:
            apply_blur_region(out, expanded, mode=mode, strength=strength)
    return out
