"""路径与默认配置。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
JOBS_DIR = ROOT / "jobs"
SAMPLES_DIR = ROOT / "samples"
ENHANCE_WEIGHTS = WEIGHTS_DIR / "enhance"

DEFAULT_DEVICE = "cuda:0"
DEFAULT_MARGIN = 0.12
DEFAULT_STRENGTH = 0.75
MAX_QUEUE_DEPTH = 8
