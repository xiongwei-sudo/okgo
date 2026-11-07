"""增强模块：N1 Real-ESRGAN / N2 低光·去噪 / N3 稳像 / N4 BasicVSR++。

默认链路：稳像 → 低光 → 去噪/去糊 → 超分，再进入打码（若启用）。
官方权重可加载时走 GPU torch；否则 OpenCV 降级，并在 notes/path_report 标明。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhanceOptions:
    enable: bool = False
    stabilize: bool = False  # N3 vidstab / OpenCV
    lowlight: bool = False  # N2 Zero-DCE
    denoise: bool = False  # N2 NAFNet
    deblur: bool = False  # N2 NAFNet-Deblur
    superres: bool = False  # N1 Real-ESRGAN
    scale: int = 2  # 2 or 4；官方 x4 权重在 scale=2 时仍跑 x4 再缩回目标倍率
    keep_resolution: bool = True  # True：输出与输入同分辨率；False：允许超分放大写出
    notes: List[str] = field(default_factory=list)
    # 运行期由 EnhancePipeline 填充
    path_report: Dict[str, str] = field(default_factory=dict)


def _has_weight(weights_dir: Path, patterns: Tuple[str, ...]) -> Optional[Path]:
    root = Path(weights_dir)
    for pat in patterns:
        hits = list(root.glob(pat)) + list((root / "enhance").glob(pat))
        for h in hits:
            if h.is_file() and h.stat().st_size > 1000:
                return h
    return None


# ---------------------------------------------------------------------------
# Minimal torch models (no basicsr / realesrgan pip dependency)
# ---------------------------------------------------------------------------
def _build_rrdb_net():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # x4: two nearest-upsample + conv
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    return RRDBNet


def _build_zero_dce():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class EnhanceNet(nn.Module):
        """Original Zero-DCE (Li et al.): 7 convs → 8 curve parameter maps × 3ch = 24."""

        def __init__(self):
            super().__init__()
            self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1)
            self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
            self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
            self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1)
            self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1)
            self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1)
            self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x1 = self.relu(self.e_conv1(x))
            x2 = self.relu(self.e_conv2(x1))
            x3 = self.relu(self.e_conv3(x2))
            x4 = self.relu(self.e_conv4(x3))
            x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
            x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
            r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
            x = x + r1 * (torch.pow(x, 2) - x)
            x = x + r2 * (torch.pow(x, 2) - x)
            x = x + r3 * (torch.pow(x, 2) - x)
            enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
            x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
            x = x + r6 * (torch.pow(x, 2) - x)
            x = x + r7 * (torch.pow(x, 2) - x)
            enhance_image = x + r8 * (torch.pow(x, 2) - x)
            return enhance_image

    return EnhanceNet


class EnhancePipeline:
    """按需加载增强模型；缺权重或加载失败时 OpenCV 降级。"""

    def __init__(self, weights_dir: Path, device: str = "cuda:0"):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self._loaded: List[str] = []
        self._warned: set[str] = set()
        self._prev_gray: Optional[np.ndarray] = None
        self._transform_acc = np.eye(2, 3, dtype=np.float64)
        self._realesrgan = None
        self._realesrgan_scale = 4
        self._zero_dce = None
        self._torch_device = None
        self.path_report: Dict[str, str] = {}

    def _resolve_torch_device(self) -> str:
        try:
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                return self.device if ":" in self.device else "cuda:0"
        except Exception:
            pass
        return "cpu"

    def load_for(self, opts: EnhanceOptions) -> None:
        if not opts.enable:
            return
        self.path_report = {}
        self._torch_device = self._resolve_torch_device()
        pending: List[str] = []

        if opts.stabilize:
            # 无独立大权重
            self.path_report["stabilize"] = "opencv_optical_flow"
            pending.append("N3-稳像 → OpenCV 光流（无独立大权重）")

        if opts.lowlight:
            w = _has_weight(self.weights_dir, ("*zero_dce*", "*Zero-DCE*", "*dce*.pth"))
            if w and self._load_zero_dce(w):
                self.path_report["lowlight"] = f"official_gpu:{w.name}@{self._torch_device}"
                pending.append(f"N2-Zero-DCE 官方权重已加载 GPU/设备={self._torch_device} ({w.name})")
            else:
                self.path_report["lowlight"] = "opencv_clahe_fallback"
                pending.append("N2-Zero-DCE【降级】→ OpenCV CLAHE（权重缺失或加载失败）")

        if opts.denoise:
            w = _has_weight(self.weights_dir, ("*NAFNet*", "*nafnet*", "*denoise*"))
            self.path_report["denoise"] = "opencv_fallback"
            if w:
                pending.append(f"N2-NAFNet 权重已发现但仍【待补】接入 ({w.name}) → OpenCV 去噪降级")
            else:
                pending.append("N2-NAFNet【待补】→ OpenCV 去噪降级")

        if opts.deblur:
            w = _has_weight(self.weights_dir, ("*Deblur*", "*deblur*"))
            self.path_report["deblur"] = "opencv_sharpen_fallback"
            if w:
                pending.append(f"N2-Deblur 权重已发现但仍【待补】接入 ({w.name}) → OpenCV 锐化降级")
            else:
                pending.append("N2-Deblur【待补】→ OpenCV 锐化降级")

        if opts.superres:
            w = _has_weight(
                self.weights_dir,
                ("*RealESRGAN*", "*realesrgan*", "*ESRGAN*.pth", f"*x{opts.scale}*"),
            )
            if w and self._load_realesrgan(w):
                self.path_report["superres"] = (
                    f"official_gpu:{w.name}@x{self._realesrgan_scale}/{self._torch_device}"
                )
                pending.append(
                    f"N1-Real-ESRGAN 官方权重已加载 ({w.name}, native_x{self._realesrgan_scale}, "
                    f"device={self._torch_device}, keep_resolution={opts.keep_resolution})"
                )
            else:
                self.path_report["superres"] = "opencv_resize_sharpen_fallback"
                pending.append(
                    f"N1-Real-ESRGAN-x{opts.scale}【降级】→ OpenCV 同分辨率增强（权重缺失或加载失败）"
                )

        opts.path_report = dict(self.path_report)
        for p in pending:
            if p not in opts.notes:
                opts.notes.append(p)
            if p not in self._warned:
                logger.info("%s", p)
                self._warned.add(p)
        self._loaded = pending

        # 汇总一行便于 UI
        summary = ", ".join(f"{k}={v}" for k, v in self.path_report.items()) or "none"
        opts.notes.append(f"enhance_path: {summary}")
        logger.info("enhance_path: %s", summary)

    def _load_zero_dce(self, weight: Path) -> bool:
        try:
            import torch

            EnhanceNet = _build_zero_dce()
            model = EnhanceNet()
            sd = torch.load(str(weight), map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and any(k.startswith("e_conv") for k in sd):
                model.load_state_dict(sd, strict=True)
            elif isinstance(sd, dict) and "params" in sd:
                model.load_state_dict(sd["params"], strict=True)
            else:
                model.load_state_dict(sd, strict=False)
            device = self._torch_device or "cpu"
            model.eval().to(device)
            self._zero_dce = model
            logger.info("Zero-DCE loaded from %s on %s", weight, device)
            return True
        except Exception as e:
            logger.warning("Zero-DCE load failed: %s", e)
            self._zero_dce = None
            return False

    def _load_realesrgan(self, weight: Path) -> bool:
        try:
            import torch

            RRDBNet = _build_rrdb_net()
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
            ckpt = torch.load(str(weight), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "params_ema" in ckpt:
                sd = ckpt["params_ema"]
            elif isinstance(ckpt, dict) and "params" in ckpt:
                sd = ckpt["params"]
            else:
                sd = ckpt
            model.load_state_dict(sd, strict=True)
            device = self._torch_device or "cpu"
            model.eval().to(device)
            self._realesrgan = model
            self._realesrgan_scale = 4
            logger.info("RealESRGAN loaded from %s on %s", weight, device)
            return True
        except Exception as e:
            logger.warning("RealESRGAN load failed: %s", e)
            self._realesrgan = None
            return False

    def unload(self) -> None:
        self._loaded = []
        self._prev_gray = None
        self._transform_acc = np.eye(2, 3, dtype=np.float64)
        self._warned.clear()
        self.path_report = {}
        for attr in ("_realesrgan", "_zero_dce", "_nafnet", "_basicvsr"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if self._prev_gray is None:
            self._prev_gray = gray
            return frame
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )
        if prev_pts is None or len(prev_pts) < 8:
            self._prev_gray = gray
            return frame
        nxt_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
        if nxt_pts is None or status is None:
            self._prev_gray = gray
            return frame
        ok_prev = prev_pts[status.flatten() == 1]
        ok_next = nxt_pts[status.flatten() == 1]
        if len(ok_prev) < 8:
            self._prev_gray = gray
            return frame
        m, _inliers = cv2.estimateAffinePartial2D(ok_prev, ok_next)
        if m is None:
            self._prev_gray = gray
            return frame
        try:
            inv = cv2.invertAffineTransform(m)
        except cv2.error:
            self._prev_gray = gray
            return frame
        self._transform_acc[:, :2] = 0.9 * self._transform_acc[:, :2] + 0.1 * inv[:, :2]
        self._transform_acc[:, 2] = 0.9 * self._transform_acc[:, 2] + 0.1 * inv[:, 2]
        stabilized = cv2.warpAffine(
            frame,
            self._transform_acc,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        self._prev_gray = gray
        return stabilized

    def _lowlight_opencv(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    def _lowlight_zero_dce(self, frame: np.ndarray) -> np.ndarray:
        import torch

        assert self._zero_dce is not None
        device = self._torch_device or "cpu"
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self._zero_dce(t)
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if h * w > 1280 * 720:
            return cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)
        return cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)

    def _deblur(self, frame: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.2)
        return cv2.addWeighted(frame, 1.6, blur, -0.6, 0)

    def _superres_opencv(self, frame: np.ndarray, scale: int) -> np.ndarray:
        h, w = frame.shape[:2]
        s = max(2, int(scale))
        up = cv2.resize(frame, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
        sharp = self._deblur(up)
        return cv2.resize(sharp, (w, h), interpolation=cv2.INTER_AREA)

    def _superres_realesrgan(self, frame: np.ndarray, opts: EnhanceOptions) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        assert self._realesrgan is not None
        device = self._torch_device or "cpu"
        h0, w0 = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        # 大图分块推理，避免 OOM；tile=0 表示整图
        tile = 400 if max(h0, w0) > 720 else 0
        overlap = 16
        with torch.no_grad():
            if tile <= 0:
                out = self._realesrgan(t)
            else:
                out = self._tile_sr(t, tile=tile, overlap=overlap)

        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        bgr = cv2.cvtColor((out * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        target_scale = max(2, int(opts.scale))
        native = self._realesrgan_scale  # 4
        if opts.keep_resolution:
            return cv2.resize(bgr, (w0, h0), interpolation=cv2.INTER_AREA)
        # 目标倍率：若请求 2x 而模型 4x，缩到 2x
        if target_scale != native:
            nh, nw = h0 * target_scale, w0 * target_scale
            return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return bgr

    def _tile_sr(self, img, tile: int, overlap: int):
        import torch
        import torch.nn.functional as F

        b, c, h, w = img.shape
        scale = self._realesrgan_scale
        out_h, out_w = h * scale, w * scale
        output = img.new_zeros((b, c, out_h, out_w))
        weights = img.new_zeros((b, 1, out_h, out_w))
        stride = max(1, tile - overlap)
        ys = list(range(0, h, stride))
        xs = list(range(0, w, stride))
        if not ys:
            ys = [0]
        if not xs:
            xs = [0]
        for y in ys:
            for x in xs:
                y1, x1 = y, x
                y2, x2 = min(y + tile, h), min(x + tile, w)
                # 扩展到至少能走网络
                if y2 - y1 < 16 or x2 - x1 < 16:
                    continue
                patch = img[:, :, y1:y2, x1:x2]
                out = self._realesrgan(patch)
                oy1, ox1 = y1 * scale, x1 * scale
                oy2, ox2 = y2 * scale, x2 * scale
                output[:, :, oy1:oy2, ox1:ox2] += out
                weights[:, :, oy1:oy2, ox1:ox2] += 1.0
        weights = torch.clamp(weights, min=1.0)
        return output / weights

    def process_frame(self, frame: np.ndarray, opts: EnhanceOptions) -> np.ndarray:
        if not opts.enable:
            return frame
        out = frame
        if opts.stabilize:
            out = self._stabilize_frame(out)
        if opts.lowlight:
            if self._zero_dce is not None:
                out = self._lowlight_zero_dce(out)
            else:
                out = self._lowlight_opencv(out)
        if opts.denoise:
            out = self._denoise(out)
        if opts.deblur:
            out = self._deblur(out)
        if opts.superres:
            if self._realesrgan is not None:
                out = self._superres_realesrgan(out, opts)
            else:
                out = self._superres_opencv(out, opts.scale)
        return out

    def process_video_path(
        self, input_path: Path, output_path: Path, opts: EnhanceOptions
    ) -> Optional[Path]:
        if not opts.enable:
            return None
        self.load_for(opts)
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error("EnhancePipeline 无法打开: %s", input_path)
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if opts.superres and not opts.keep_resolution and self._realesrgan is not None:
            s = max(2, int(opts.scale))
            out_w, out_h = width * s, height * s
        else:
            out_w, out_h = width, height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            logger.error("EnhancePipeline VideoWriter 失败")
            return None
        self._prev_gray = None
        self._transform_acc = np.eye(2, 3, dtype=np.float64)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out = self.process_frame(frame, opts)
                if out.shape[1] != out_w or out.shape[0] != out_h:
                    out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(out)
        finally:
            cap.release()
            writer.release()
        logger.info("EnhancePipeline wrote %s path=%s", output_path, self.path_report)
        return output_path
