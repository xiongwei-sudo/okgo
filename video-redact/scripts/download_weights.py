#!/usr/bin/env python3
"""下载人脸 YuNet，并按公开地址尝试车牌 YOLOv8 权重。

车牌全部失败时仍会刷新 weights/PLATE_WEIGHTS_PENDING.txt。
检测器保持颜色/轮廓启发式，不在这里中断。OCR 由 RapidOCR 首次运行自拉。
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://hf-mirror.com/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]

# 保存为 weights/plate_yolov8n.pt。这些是公开 YOLOv8 车牌检测权重，任一成功即可。
PLATE_CANDIDATES = [
    (
        "plate_yolov8n.pt",
        [
            "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
            "https://hf-mirror.com/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
            "https://huggingface.co/joker5914/yolov8n-license-plate/resolve/main/best.pt",
            "https://hf-mirror.com/joker5914/yolov8n-license-plate/resolve/main/best.pt",
            "https://huggingface.co/OpenPathAI/YOLO-detection-vehcile-plate-2287/resolve/main/weights/best.pt",
            "https://hf-mirror.com/OpenPathAI/YOLO-detection-vehcile-plate-2287/resolve/main/weights/best.pt",
        ],
    ),
]

PLATE_NAMES = (
    "plate_yolov8n.pt",
    "yolov8n-plate.pt",
    "license_plate_yolov8n.pt",
)


def file_ok(path: Path, kind: str) -> bool:
    """拒绝 HTML/JSON 错误页和 Git LFS 指针。不把整文件读入内存。"""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < 1000:
        return False
    try:
        with path.open("rb") as handle:
            head = handle.read(256)
    except OSError:
        return False
    sample = head.lstrip().lower()
    if sample.startswith((b"<", b"{", b"not found", b"entry not", b"version https://git-lfs")):
        return False
    if b"<html" in sample or b"<!doctype" in sample:
        return False
    if kind == "pt":
        if size < 500_000:
            return False
        if not (head.startswith(b"PK") or head[:1] == b"\x80"):
            return False
    return True


def plate_pending_text() -> str:
    lines = [
        "PLATE_WEIGHTS_PENDING",
        "",
        "车牌检测优先加载 YOLOv8n 权重。下列任一文件存在且通过校验时走 YOLO，",
        "否则状态为【待补】，检测器继续使用蓝/绿牌颜色与轮廓启发式，不中断打码：",
        "",
    ]
    lines.extend(f"- {name}" for name in PLATE_NAMES)
    lines.extend(
        [
            "",
            "获取方式：",
            "  python scripts/download_weights.py",
            "",
            "脚本会按顺序尝试这些公开地址（成功一个即停止，保存为 plate_yolov8n.pt）：",
        ]
    )
    for _name, urls in PLATE_CANDIDATES:
        lines.extend(f"- {url}" for url in urls)
    lines.extend(
        [
            "",
            "权重 *.pt / *.onnx 已在 .gitattributes 中标记 Git LFS，",
            "同时被 .gitignore 排除。不要把大权重提交进博客仓库。",
            "工作区里没有现成的小型 YuNet onnx 时，也不要为了入库去下载它。",
            "",
        ]
    )
    return "\n".join(lines)


def download(url: str, dest: Path, kind: str, timeout: int = 120) -> bool:
    print(f"GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "video-redact/0.2"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        if not file_ok(tmp, kind):
            size = tmp.stat().st_size if tmp.exists() else 0
            print(f"  reject ({size} bytes), skip")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        print(f"  -> {dest} ({dest.stat().st_size} bytes)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    yunet = WEIGHTS / "face_detection_yunet_2023mar.onnx"
    if file_ok(yunet, "onnx"):
        print(f"YuNet OK: {yunet}")
    else:
        ok = False
        for url in YUNET_URLS:
            if download(url, yunet, "onnx"):
                ok = True
                break
        if not ok:
            print("【待补】YuNet 人脸权重下载失败", file=sys.stderr)

    plate_ok = False
    for name, urls in PLATE_CANDIDATES:
        dest = WEIGHTS / name
        if file_ok(dest, "pt"):
            print(f"Plate OK: {dest}")
            plate_ok = True
            break
        for url in urls:
            if download(url, dest, "pt"):
                plate_ok = True
                break
        if plate_ok:
            break
    if not plate_ok:
        for alias in PLATE_NAMES:
            path = WEIGHTS / alias
            if file_ok(path, "pt"):
                plate_ok = True
                print(f"Plate OK: {path}")
                break

    note = WEIGHTS / "PLATE_WEIGHTS_PENDING.txt"
    note.write_text(plate_pending_text(), encoding="utf-8")
    if plate_ok:
        print("车牌权重已就绪。说明仍保留在 PLATE_WEIGHTS_PENDING.txt")
    else:
        print("【待补】车牌权重未就绪，已写 PLATE_WEIGHTS_PENDING.txt；启发式降级仍然可用")

    enh = WEIGHTS / "enhance"
    enh.mkdir(exist_ok=True)
    (enh / "README.txt").write_text(
        "N1 Real-ESRGAN / N2 NAFNet·Zero-DCE++ / N3 vidstab / N4 BasicVSR++ 权重【待补】\n"
        "缺失时流水线使用 OpenCV：稳像特征点、低光 gamma/CLAHE、去噪 fastNlMeans、"
        "去糊反锐化、超分 INTER_CUBIC，并在日志中保留【待补】。\n",
        encoding="utf-8",
    )
    print("Done. weights dir:", WEIGHTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
