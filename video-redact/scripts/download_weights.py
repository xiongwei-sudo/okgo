#!/usr/bin/env python3
"""下载人脸 YuNet、尝试车牌 YOLOv8n 权重；OCR 由 RapidOCR 首次运行自拉。"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

YUNET_URLS = [
    # OpenCV Zoo YuNet
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://hf-mirror.com/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]

# 常见开源车牌权重候选（任一成功即可）；全失败则【待补】
PLATE_CANDIDATES = [
    (
        "plate_yolov8n.pt",
        [
            # keremberke HF / github releases 等公开权重；镜像多源尝试
            "https://github.com/akanametov/yolo-plate/releases/download/v0.0.1/yolov8n.pt",
            "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/best.pt",
            "https://hf-mirror.com/keremberke/yolov8n-license-plate/resolve/main/best.pt",
        ],
    ),
]


def download(url: str, dest: Path, timeout: int = 120) -> bool:
    print(f"GET {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "video-redact/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1000:
            print(f"  too small ({len(data)} bytes), skip")
            return False
        dest.write_bytes(data)
        print(f"  -> {dest} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main() -> int:
    yunet = WEIGHTS / "face_detection_yunet_2023mar.onnx"
    if yunet.exists() and yunet.stat().st_size > 1000:
        print(f"YuNet OK: {yunet}")
    else:
        ok = False
        for u in YUNET_URLS:
            if download(u, yunet):
                ok = True
                break
        if not ok:
            print("【待补】YuNet 人脸权重下载失败", file=sys.stderr)

    plate_ok = False
    for name, urls in PLATE_CANDIDATES:
        dest = WEIGHTS / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"Plate OK: {dest}")
            plate_ok = True
            break
        for u in urls:
            if download(u, dest):
                plate_ok = True
                break
        if plate_ok:
            break
    if not plate_ok:
        note = WEIGHTS / "PLATE_WEIGHTS_PENDING.txt"
        note.write_text(
            "车牌 YOLOv8n 微调权重【待补】\n"
            "请将 plate_yolov8n.pt 放到本目录。\n"
            "流水线会自动降级为颜色/轮廓启发式。\n",
            encoding="utf-8",
        )
        print("【待补】车牌权重未就绪，已写 PLATE_WEIGHTS_PENDING.txt")

    # 占位：增强权重目录
    enh = WEIGHTS / "enhance"
    enh.mkdir(exist_ok=True)
    (enh / "README.txt").write_text(
        "N1 Real-ESRGAN / N2 NAFNet·Zero-DCE++ / N3 vidstab / N4 BasicVSR++ 权重【待补】\n",
        encoding="utf-8",
    )
    print("Done. weights dir:", WEIGHTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
