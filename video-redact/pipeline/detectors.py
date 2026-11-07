"""检测器：人脸 / 车牌 / OCR。按需加载，用完可卸载。"""
from __future__ import annotations

import logging
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]
Detection = Tuple[BBox, str, float]  # box, label, score

ID_CARD_RE = re.compile(r"\d{17}[\dXx]")
PHONE_RE = re.compile(r"1[3-9]\d{9}")
SENSITIVE_RES = (ID_CARD_RE, PHONE_RE)


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]: ...

    @property
    @abstractmethod
    def loaded(self) -> bool: ...


class FaceDetector(BaseDetector):
    """优先 MediaPipe；失败则 YuNet（OpenCV Zoo）。"""

    name = "face"

    def __init__(self, weights_dir: Path, conf: float = 0.5, device: str = "cpu"):
        self.weights_dir = Path(weights_dir)
        self.conf = conf
        self.device = device
        self._mp = None
        self._yunet = None
        self._backend = None

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import mediapipe as mp

            self._mp = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=self.conf
            )
            self._backend = "mediapipe"
            logger.info("FaceDetector loaded: MediaPipe")
            return
        except Exception as e:
            logger.warning("MediaPipe face failed (%s), trying YuNet", e)

        yunet = self.weights_dir / "face_detection_yunet_2023mar.onnx"
        if not yunet.exists():
            raise FileNotFoundError(f"YuNet weights missing: {yunet}")
        self._yunet = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), self.conf, 0.3, 5000)
        self._backend = "yunet"
        logger.info("FaceDetector loaded: YuNet %s", yunet)

    def unload(self) -> None:
        if self._mp is not None:
            try:
                self._mp.close()
            except Exception:
                pass
        self._mp = None
        self._yunet = None
        self._backend = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.loaded:
            self.load()
        h, w = frame.shape[:2]
        out: List[Detection] = []
        if self._backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._mp.process(rgb)
            if res.detections:
                for det in res.detections:
                    score = float(det.score[0]) if det.score else 0.0
                    if score < self.conf:
                        continue
                    bb = det.location_data.relative_bounding_box
                    x1 = int(bb.xmin * w)
                    y1 = int(bb.ymin * h)
                    x2 = int((bb.xmin + bb.width) * w)
                    y2 = int((bb.ymin + bb.height) * h)
                    out.append(((x1, y1, x2, y2), "face", score))
        elif self._backend == "yunet":
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(frame)
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = map(int, f[:4])
                    score = float(f[-1])
                    if score < self.conf:
                        continue
                    out.append(((x, y, x + bw, y + bh), "face", score))
        return out


class PlateDetector(BaseDetector):
    """YOLOv8 车牌；无权重时启发式颜色轮廓降级。"""

    name = "plate"

    def __init__(self, weights_dir: Path, conf: float = 0.35, device: str = "cuda:0"):
        self.weights_dir = Path(weights_dir)
        self.conf = conf
        self.device = device
        self._model = None
        self._backend = None
        self.weight_path: Optional[Path] = None

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def _find_weight(self) -> Optional[Path]:
        candidates = [
            self.weights_dir / "plate_yolov8n.pt",
            self.weights_dir / "yolov8n-plate.pt",
            self.weights_dir / "license_plate_yolov8n.pt",
        ]
        for p in candidates:
            if p.exists() and p.stat().st_size > 1000:
                return p
        return None

    def load(self) -> None:
        if self.loaded:
            return
        w = self._find_weight()
        if w is not None:
            from ultralytics import YOLO

            self._model = YOLO(str(w))
            self.weight_path = w
            self._backend = "yolo"
            logger.info("PlateDetector loaded: YOLO %s", w)
        else:
            self._backend = "heuristic"
            logger.warning("车牌权重【待补】，使用颜色/轮廓启发式降级")

    def unload(self) -> None:
        self._model = None
        self._backend = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.loaded:
            self.load()
        out: List[Detection] = []
        if self._backend == "yolo":
            results = self._model.predict(
                frame, conf=self.conf, verbose=False, device=self.device
            )
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    score = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, xyxy)
                    out.append(((x1, y1, x2, y2), "plate", score))
        else:
            out.extend(self._heuristic(frame))
        return out

    def _heuristic(self, frame: np.ndarray) -> List[Detection]:
        """粗略蓝牌/绿牌区域；仅作降级占位。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = []
        # 蓝牌
        masks.append(cv2.inRange(hsv, (100, 80, 80), (130, 255, 255)))
        # 绿牌
        masks.append(cv2.inRange(hsv, (35, 40, 40), (90, 255, 255)))
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        out: List[Detection] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < 800 or area > w * h * 0.25:
                continue
            ratio = bw / max(bh, 1)
            if ratio < 1.5 or ratio > 6.5:
                continue
            out.append(((x, y, x + bw, y + bh), "plate", 0.3))
        return out[:20]


class TextDetector(BaseDetector):
    """RapidOCR：检测 + 识别，仅对敏感模式打码。"""

    name = "text"

    def __init__(self, conf: float = 0.5, sensitive_only: bool = True):
        self.conf = conf
        self.sensitive_only = sensitive_only
        self._ocr = None

    @property
    def loaded(self) -> bool:
        return self._ocr is not None

    def load(self) -> None:
        if self.loaded:
            return
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR()
        logger.info("TextDetector loaded: RapidOCR")

    def unload(self) -> None:
        self._ocr = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.loaded:
            self.load()
        result, _ = self._ocr(frame)
        out: List[Detection] = []
        if not result:
            return out
        for item in result:
            # item: [box(4 points), text, score]
            box_pts, text, score = item[0], item[1], float(item[2])
            if score < self.conf:
                continue
            text = text or ""
            if self.sensitive_only and not any(r.search(text) for r in SENSITIVE_RES):
                continue
            xs = [p[0] for p in box_pts]
            ys = [p[1] for p in box_pts]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            out.append(((x1, y1, x2, y2), "text", score))
        return out


class DetectorHub:
    """按需加载；一任务结束可 unload_all。线程不安全：一卡一任务假定单 worker。"""

    def __init__(self, weights_dir: Path, device: str = "cuda:0"):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.face = FaceDetector(self.weights_dir, device=device)
        self.plate = PlateDetector(self.weights_dir, device=device)
        self.text = TextDetector()
        self._lock = threading.Lock()

    def get(self, kinds: Sequence[str]) -> List[BaseDetector]:
        mapping = {"face": self.face, "plate": self.plate, "text": self.text}
        dets = []
        for k in kinds:
            if k in mapping:
                det = mapping[k]
                if not det.loaded:
                    det.load()
                dets.append(det)
        return dets

    def unload_all(self) -> None:
        for d in (self.face, self.plate, self.text):
            try:
                d.unload()
            except Exception as e:
                logger.warning("unload %s: %s", d.name, e)
