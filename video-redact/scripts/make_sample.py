#!/usr/bin/env python3
"""生成合成样片：含假人脸色块、蓝牌矩形、身份证号字幕。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=Path("samples/synthetic_smoke.mp4"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", type=int, nargs=2, default=[640, 360])
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    w, h = args.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (w, h))
    for i in range(args.frames):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        # 移动「脸」椭圆
        cx = int(120 + i * 3) % (w - 80)
        cy = 100
        cv2.ellipse(frame, (cx, cy), (35, 45), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(frame, (cx - 12, cy - 10), 5, (30, 30, 30), -1)
        cv2.circle(frame, (cx + 12, cy - 10), 5, (30, 30, 30), -1)
        # 蓝牌
        px = int(200 + (i % 30))
        py = 220
        cv2.rectangle(frame, (px, py), (px + 160, py + 40), (200, 80, 20), -1)
        cv2.putText(frame, "京A12345", (px + 10, py + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # 敏感文字
        cv2.putText(
            frame,
            "ID:110101199001011234 phone:13800138000",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )
        writer.write(frame)
    writer.release()
    print("wrote", args.output)


if __name__ == "__main__":
    main()
