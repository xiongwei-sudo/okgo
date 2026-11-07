#!/usr/bin/env python3
"""打印关键依赖版本，供验收报告。"""
from __future__ import annotations

def main() -> None:
    import importlib
    pairs = [
        ("torch", "torch"),
        ("cv2", "opencv-python"),
        ("ultralytics", "ultralytics"),
        ("mediapipe", "mediapipe"),
        ("rapidocr_onnxruntime", "rapidocr"),
        ("onnxruntime", "onnxruntime"),
        ("fastapi", "fastapi"),
        ("numpy", "numpy"),
    ]
    for mod, label in pairs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{label}: {ver}")
        except Exception as e:
            print(f"{label}: MISSING ({e})")
    try:
        import torch
        print(f"torch.cuda: {torch.version.cuda} avail={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
            x = torch.zeros(1, device="cuda")
            print(f"cuda_tensor_ok: {x.device}")
    except Exception as e:
        print("cuda_check:", e)


if __name__ == "__main__":
    main()
