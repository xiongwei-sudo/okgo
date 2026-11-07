#!/usr/bin/env python3
"""CLI：视频打码入口。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.enhance import EnhanceOptions
from pipeline.pipeline import RedactOptions, run_redact

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="视频打码 CLI（人脸/车牌/OCR）")
    p.add_argument("-i", "--input", required=True, help="输入视频路径")
    p.add_argument("-o", "--output", required=True, help="输出视频路径")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--no-plate", action="store_true")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--mode", choices=["gaussian", "mosaic", "solid"], default="gaussian")
    p.add_argument("--strength", type=float, default=0.75)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    p.add_argument("--enhance", action="store_true", help="启用增强；官方权重缺失时走 OpenCV 降级并记录【待补】")
    p.add_argument("--superres", action="store_true")
    p.add_argument("--lowlight", action="store_true")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--deblur", action="store_true")
    p.add_argument("--stabilize", action="store_true")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = "cpu" if args.cpu else args.device
    enhance = EnhanceOptions(
        enable=args.enhance
        or args.superres
        or args.lowlight
        or args.denoise
        or args.deblur
        or args.stabilize,
        stabilize=args.stabilize,
        lowlight=args.lowlight,
        denoise=args.denoise,
        deblur=args.deblur,
        superres=args.superres,
        scale=args.scale,
    )
    opts = RedactOptions(
        face=not args.no_face,
        plate=not args.no_plate,
        text=not args.no_text,
        mode=args.mode,
        strength=args.strength,
        stride=args.stride,
        device=device,
        enhance=enhance,
    )
    result = run_redact(Path(args.input), Path(args.output), Path(args.weights), opts)
    print(f"OK frames={result.frames} detections={result.detections_total} -> {result.output}")
    for n in result.notes:
        print(f"NOTE: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
