"""简易 IoU 跟踪，减轻漏检闪烁。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

BBox = Tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: BBox
    miss: int = 0
    hits: int = 1


@dataclass
class SimpleTracker:
    iou_thresh: float = 0.3
    max_miss: int = 5
    _next_id: int = 1
    tracks: List[Track] = field(default_factory=list)

    def update(self, detections: Sequence[BBox]) -> List[BBox]:
        dets = [tuple(map(float, d)) for d in detections]
        assigned = set()
        for tr in self.tracks:
            best_j, best_iou = -1, 0.0
            for j, d in enumerate(dets):
                if j in assigned:
                    continue
                v = iou(tr.bbox, d)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                tr.bbox = dets[best_j]
                tr.miss = 0
                tr.hits += 1
                assigned.add(best_j)
            else:
                tr.miss += 1

        for j, d in enumerate(dets):
            if j not in assigned:
                self.tracks.append(Track(track_id=self._next_id, bbox=d))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        return [t.bbox for t in self.tracks if t.miss == 0 or t.hits >= 2]
