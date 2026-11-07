#!/usr/bin/env python3
"""下载人脸 YuNet、尝试多源车牌 YOLO 权重；OCR 由 RapidOCR 首次运行自拉。"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://cdn.jsdelivr.net/gh/opencv/opencv_zoo@main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://hf-mirror.com/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]

# 仅车牌相关公开候选（任一成功即可）；全失败则写 PLATE_WEIGHTS_PENDING.txt
PLATE_CANDIDATES = [
    (
        "plate_yolov8n.pt",
        [
            # keremberke YOLOv8n license-plate（HF + 镜像 + 常见文件名）
            "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/best.pt",
            "https://hf-mirror.com/keremberke/yolov8n-license-plate/resolve/main/best.pt",
            "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/yolov8n-license-plate.pt",
            "https://hf-mirror.com/keremberke/yolov8n-license-plate/resolve/main/yolov8n-license-plate.pt",
            "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/last.pt",
            "https://hf-mirror.com/keremberke/yolov8n-license-plate/resolve/main/last.pt",
            # 社区 release
            "https://github.com/akanametov/yolo-plate/releases/download/v0.0.1/yolov8n.pt",
            "https://github.com/akanametov/yolo-plate/releases/download/v0.0.1/plate_yolov8n.pt",
            "https://github.com/kaylode/yolo-plate/releases/download/v0.1/yolov8n-plate.pt",
            "https://github.com/kaylode/yolo-plate/releases/download/v0.1/best.pt",
            # HF 其它车牌微调（尽力）
            "https://huggingface.co/spaces/keremberke/yolov8-license-plate-detection/resolve/main/best.pt",
            "https://hf-mirror.com/spaces/keremberke/yolov8-license-plate-detection/resolve/main/best.pt",
            "https://huggingface.co/Bingsu/yolo-license-plate/resolve/main/best.pt",
            "https://hf-mirror.com/Bingsu/yolo-license-plate/resolve/main/best.pt",
        ],
    ),
    (
        "plate_yolov5.pt",
        [
            "https://github.com/wppurking/yolov5_plate/releases/download/v1.0/best.pt",
            "https://github.com/kadirnar/yolov5-license-plate/releases/download/v0.1/best.pt",
            "https://huggingface.co/keremberke/yolov5n-license-plate/resolve/main/best.pt",
            "https://hf-mirror.com/keremberke/yolov5n-license-plate/resolve/main/best.pt",
        ],
    ),
]


def download(url: str, dest: Path, timeout: int = 180) -> bool:
    print(f"GET {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "video-redact/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 50_000:
            head = data[:200].lstrip().lower()
            if head.startswith(b"<!doctype") or head.startswith(b"<html") or b"not found" in head:
                print(f"  looks like HTML ({len(data)} bytes), skip")
                return False
            print(f"  too small ({len(data)} bytes), skip")
            return False
        dest.write_bytes(data)
        print(f"  -> {dest} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def write_plate_pending() -> None:
    note = WEIGHTS / "PLATE_WEIGHTS_PENDING.txt"
    note.write_text(
        "车牌 YOLO 微调权重【待补】\n"
        "请将 plate_yolov8n.pt（或兼容 ultralytics .pt）放到本目录。\n"
        "流水线会自动降级为颜色/轮廓启发式。\n"
        "已尝试多源公开 URL；仍失败时保留本文件作为标记。\n",
        encoding="utf-8",
    )
    print("【待补】车牌权重未就绪，已写 PLATE_WEIGHTS_PENDING.txt")


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
        if dest.exists() and dest.stat().st_size > 50_000:
            print(f"Plate OK: {dest}")
            plate_ok = True
            break
        for u in urls:
            if download(u, dest):
                plate_ok = True
                break
        if plate_ok:
            break

    pending = WEIGHTS / "PLATE_WEIGHTS_PENDING.txt"
    if plate_ok:
        if pending.exists():
            pending.unlink()
        print("车牌权重就绪，已移除 PLATE_WEIGHTS_PENDING.txt（若存在）")
    else:
        write_plate_pending()

    enh = WEIGHTS / "enhance"
    enh.mkdir(exist_ok=True)
    (enh / "README.txt").write_text(
        "N1 Real-ESRGAN / N2 NAFNet·Zero-DCE++ / N3 vidstab / N4 BasicVSR++ 权重【待补】\n"
        "缺权重时 pipeline/enhance.py 使用 OpenCV 降级。\n",
        encoding="utf-8",
    )
    print("Done. weights dir:", WEIGHTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
