"""OpenCV 增强降级、车牌说明，以及超分会改变输出尺寸。"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.detectors import BaseDetector, DetectorHub, PlateDetector  # noqa: E402
from pipeline.enhance import EnhanceOptions, EnhancePipeline  # noqa: E402
from pipeline.pipeline import RedactOptions, run_redact  # noqa: E402


def _load_download_mod():
    path = ROOT / "scripts" / "download_weights.py"
    spec = importlib.util.spec_from_file_location("download_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EnhanceFallbackTest(unittest.TestCase):
    def test_opencv_fallbacks_log_pending_and_scale(self) -> None:
        frame = np.full((32, 40, 3), 24, np.uint8)
        frame[8:20, 10:30] = 180
        opts = EnhanceOptions(
            enable=True,
            stabilize=True,
            lowlight=True,
            denoise=True,
            deblur=True,
            superres=True,
            scale=2,
        )
        pipe = EnhancePipeline(ROOT / "weights")
        first = pipe.process_frame(frame, opts)
        second = pipe.process_frame(frame, opts)
        self.assertEqual(first.shape, (64, 80, 3))
        self.assertEqual(second.shape, (64, 80, 3))
        self.assertGreater(float(first.mean()), float(frame.mean()))
        joined = "\n".join(opts.notes)
        for token in (
            "N1-Real-ESRGAN",
            "N2-Zero-DCE++",
            "N2-NAFNet",
            "N2-NAFNet-Deblur",
            "N3-vidstab",
            "【待补】",
            "INTER_CUBIC",
            "fastNlMeans",
            "gamma/CLAHE",
        ):
            self.assertIn(token, joined)
        again = pipe.process_frame(frame, opts)
        self.assertEqual(again.shape, first.shape)
        self.assertEqual(joined.count("N1-Real-ESRGAN"), 1)

    def test_weight_file_without_runtime_still_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp)
            blob = weights / "enhance"
            blob.mkdir()
            (blob / "RealESRGAN_x2plus.pth").write_bytes(b"PK" + b"x" * 2000)
            opts = EnhanceOptions(enable=True, superres=True, scale=2)
            pipe = EnhancePipeline(weights)
            pipe.load_for(opts)
            self.assertEqual(pipe.backend.get("superres"), "cubic")
            self.assertTrue(any("已在" in note and "【待补】" in note for note in opts.notes))

    def test_vidstab_prepass_hook(self) -> None:
        class _Fake:
            def stabilize(self, input_path: str, output_path: str) -> None:
                Path(output_path).write_bytes(Path(input_path).read_bytes())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "in.mp4"
            src.write_bytes(b"v" * 2000)
            opts = EnhanceOptions(enable=True, stabilize=True)
            pipe = EnhancePipeline(root)
            with patch("pipeline.enhance._load_vidstab_class", return_value=_Fake):
                pipe.load_for(opts)
                out = pipe.consume_stabilize_prepass(src, root, opts)
            self.assertEqual(out, root / "stabilized.mp4")
            self.assertEqual(pipe.backend.get("stabilize"), "vidstab-done")
            self.assertTrue(any("【待补】" in note for note in opts.notes))

    def test_superres_changes_video_size(self) -> None:
        class _Det(BaseDetector):
            name = "face"

            @property
            def loaded(self) -> bool:
                return True

            def load(self) -> None:
                return None

            def unload(self) -> None:
                return None

            def detect(self, frame):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "in.mp4"
            writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (16, 16))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((16, 16, 3), 40, np.uint8))
            writer.release()
            dst = root / "out.mp4"

            def _get(self, kinds):
                return [_Det()]

            with patch.object(DetectorHub, "get", _get):
                result = run_redact(
                    src,
                    dst,
                    root,
                    RedactOptions(
                        face=True,
                        plate=False,
                        text=False,
                        device="cpu",
                        keep_audio=False,
                        enhance=EnhanceOptions(enable=True, superres=True, scale=2),
                    ),
                )
            cap = cv2.VideoCapture(str(result.output))
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (32, 32))
            self.assertTrue(any("【待补】" in note for note in result.notes))


class PlatePendingTest(unittest.TestCase):
    def test_document_matches_downloader(self) -> None:
        module = _load_download_mod()
        text = (ROOT / "weights" / "PLATE_WEIGHTS_PENDING.txt").read_text(encoding="utf-8")
        self.assertEqual(text, module.plate_pending_text())
        self.assertIn("PLATE_WEIGHTS_PENDING", text)
        self.assertIn("Koushim/yolov8-license-plate-detection", text)
        self.assertIn("启发式", text)
        self.assertFalse(module.file_ok(ROOT / "weights" / "PLATE_WEIGHTS_PENDING.txt", "pt"))

    def test_heuristic_still_finds_blue_plate(self) -> None:
        frame = np.zeros((200, 400, 3), np.uint8)
        frame[80:120, 100:260] = (200, 40, 20)
        found = PlateDetector(ROOT / "weights")._heuristic(frame)
        self.assertTrue(found)
        self.assertEqual(found[0][1], "plate")


if __name__ == "__main__":
    unittest.main()
