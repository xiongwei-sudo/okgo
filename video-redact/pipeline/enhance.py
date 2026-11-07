"""增强：优先按需加载 N1–N4 官方权重；缺失时用 OpenCV 降级并记录【待补】。

建议顺序：稳像 → 低光 → 去噪 → 去糊 → 超分，再进入打码。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhanceOptions:
    enable: bool = False
    stabilize: bool = False  # N3 vidstab / OpenCV
    lowlight: bool = False  # N2 Zero-DCE++
    denoise: bool = False  # N2 NAFNet
    deblur: bool = False  # N2 NAFNet-Deblur
    superres: bool = False  # N1 Real-ESRGAN
    scale: int = 2  # 2 or 4
    notes: List[str] = field(default_factory=list)


def _load_vidstab_class():
    """按需导入 vidstab。未安装时抛出，由调用方改走 OpenCV。"""
    from vidstab import VidStab

    return VidStab


class EnhancePipeline:
    """按需加载增强模型。官方权重或运行时缺失时走轻量 OpenCV，并打【待补】日志。"""

    def __init__(self, weights_dir: Path, device: str = "cuda:0"):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.backend: dict[str, str] = {}
        self._sig: Optional[tuple] = None
        self._own_notes: List[str] = []
        self._upsampler = None
        self._zero_dce = None
        self._naf_denoise = None
        self._naf_deblur = None
        self._prev_gray: Optional[np.ndarray] = None
        self._traj: Optional[np.ndarray] = None
        self._smooth: Optional[np.ndarray] = None
        self._clahe = None
        self._gamma_lut: Optional[np.ndarray] = None

    def load_for(self, opts: EnhanceOptions) -> None:
        if not opts.enable:
            return
        sig = (
            opts.stabilize,
            opts.lowlight,
            opts.denoise,
            opts.deblur,
            opts.superres,
            int(opts.scale),
        )
        if sig == self._sig:
            return
        if self._own_notes:
            opts.notes[:] = [n for n in opts.notes if n not in self._own_notes]
        self._own_notes = []
        self._sig = sig
        self._reset_runtime()
        if opts.stabilize:
            self._hook_stabilize(opts)
        if opts.lowlight:
            self._hook_lowlight(opts)
        if opts.denoise:
            self._hook_denoise(opts)
        if opts.deblur:
            self._hook_deblur(opts)
        if opts.superres:
            self._hook_superres(opts)

    def unload(self) -> None:
        self._reset_runtime()
        self._sig = None
        self._own_notes = []

    def output_scale(self, opts: EnhanceOptions) -> int:
        if not opts.enable or not opts.superres:
            return 1
        return int(opts.scale) if int(opts.scale) in (2, 4) else 2

    def consume_stabilize_prepass(
        self, src: Path, work_dir: Path, opts: EnhanceOptions
    ) -> Path:
        """vidstab 可用时先整段预稳像。失败或未安装则留给逐帧 OpenCV。"""
        if not opts.enable or not opts.stabilize:
            return src
        if self.backend.get("stabilize") != "vidstab":
            return src
        dest = Path(work_dir) / "stabilized.mp4"
        try:
            stabilizer = _load_vidstab_class()()
            stabilizer.stabilize(input_path=str(src), output_path=str(dest))
        except Exception as e:
            logger.warning("vidstab 预稳像失败（%s）【待补】，改用 OpenCV 逐帧稳像", e)
            self.backend["stabilize"] = "opencv"
            return src
        if not dest.exists() or dest.stat().st_size < 1000:
            logger.warning("vidstab 未写出有效视频【待补】，改用 OpenCV 逐帧稳像")
            self.backend["stabilize"] = "opencv"
            return src
        self.backend["stabilize"] = "vidstab-done"
        logger.info("vidstab prepass -> %s", dest)
        return dest

    def process_frame(self, frame: np.ndarray, opts: EnhanceOptions) -> np.ndarray:
        if not opts.enable:
            return frame
        self.load_for(opts)
        if opts.stabilize and self.backend.get("stabilize") == "opencv":
            frame = self._stabilize_opencv(frame)
        if opts.lowlight:
            frame = self._apply_lowlight(frame)
        if opts.denoise:
            frame = self._apply_denoise(frame)
        if opts.deblur:
            frame = self._apply_deblur(frame)
        if opts.superres:
            frame = self._apply_superres(frame, self.output_scale(opts))
        return frame

    def process_video_path(
        self, input_path: Path, output_path: Path, opts: EnhanceOptions
    ) -> Optional[Path]:
        """整段入口：目前只承接 vidstab 预稳像，其余增强在 process_frame。"""
        if not opts.enable:
            return None
        self.load_for(opts)
        if not opts.stabilize:
            return None
        return self.consume_stabilize_prepass(input_path, Path(output_path).parent, opts)

    def _reset_runtime(self) -> None:
        self._upsampler = None
        self._zero_dce = None
        self._naf_denoise = None
        self._naf_deblur = None
        self._prev_gray = None
        self._traj = None
        self._smooth = None
        self._clahe = None
        self._gamma_lut = None
        self.backend = {}

    def _add_note(self, opts: EnhanceOptions, msg: str) -> None:
        logger.warning(msg)
        self._own_notes.append(msg)
        opts.notes.append(msg)

    def _find_weight(self, names: Sequence[str]) -> Optional[Path]:
        for base in (self.weights_dir, self.weights_dir / "enhance"):
            for name in names:
                path = base / name
                if path.is_file() and path.stat().st_size > 1000:
                    return path
        return None

    def _hook_stabilize(self, opts: EnhanceOptions) -> None:
        try:
            _load_vidstab_class()
            self.backend["stabilize"] = "vidstab"
            how = "vidstab 库预稳像"
        except Exception:
            self.backend["stabilize"] = "opencv"
            how = "OpenCV 特征点稳像"
        self._add_note(opts, f"N3-vidstab 权重【待补】，使用 {how}")

    def _hook_lowlight(self, opts: EnhanceOptions) -> None:
        weight = self._find_weight(("zero_dce_plus.pth", "zero_dce++.pth", "ZeroDCE++.pth"))
        if weight is not None and self._try_bind_zero_dce(weight):
            self.backend["lowlight"] = "zero_dce"
            logger.info("on-demand loaded Zero-DCE++ %s", weight)
            return
        self.backend["lowlight"] = "gamma_clahe"
        if weight is not None:
            msg = f"N2-Zero-DCE++ 权重 {weight.name} 已在，推理【待补】，使用 gamma/CLAHE"
        else:
            msg = "N2-Zero-DCE++【待补】，使用 gamma/CLAHE 低光增强"
        self._add_note(opts, msg)

    def _hook_denoise(self, opts: EnhanceOptions) -> None:
        weight = self._find_weight(("nafnet_denoise.pth", "NAFNet-SIDD-width64.pth"))
        if weight is not None and self._try_bind_nafnet(weight, "denoise"):
            self.backend["denoise"] = "nafnet"
            logger.info("on-demand loaded NAFNet denoise %s", weight)
            return
        self.backend["denoise"] = "nlmeans"
        if weight is not None:
            msg = f"N2-NAFNet 权重 {weight.name} 已在，推理【待补】，使用 fastNlMeans"
        else:
            msg = "N2-NAFNet【待补】，使用 fastNlMeans 去噪"
        self._add_note(opts, msg)

    def _hook_deblur(self, opts: EnhanceOptions) -> None:
        weight = self._find_weight(
            ("nafnet_deblur.pth", "NAFNet-REDS-width64.pth", "NAFNet-GoPro-width64.pth")
        )
        if weight is not None and self._try_bind_nafnet(weight, "deblur"):
            self.backend["deblur"] = "nafnet"
            logger.info("on-demand loaded NAFNet deblur %s", weight)
            return
        self.backend["deblur"] = "unsharp"
        if weight is not None:
            msg = f"N2-NAFNet-Deblur 权重 {weight.name} 已在，推理【待补】，使用反锐化掩模"
        else:
            msg = "N2-NAFNet-Deblur【待补】，使用反锐化掩模"
        self._add_note(opts, msg)

    def _hook_superres(self, opts: EnhanceOptions) -> None:
        scale = self.output_scale(opts)
        names = [
            f"RealESRGAN_x{scale}plus.pth",
            f"realesrgan_x{scale}.pth",
            f"RealESRGAN_x{scale}.pth",
        ]
        if scale == 4:
            names.append("realesr-general-x4v3.pth")
        weight = self._find_weight(names)
        if weight is not None and self._try_bind_realesrgan(weight, scale):
            self.backend["superres"] = "realesrgan"
            logger.info("on-demand loaded Real-ESRGAN %s", weight)
            return
        self.backend["superres"] = "cubic"
        if weight is not None:
            msg = (
                f"N1-Real-ESRGAN-x{scale} 权重 {weight.name} 已在，推理【待补】，"
                "使用 cv2.INTER_CUBIC"
            )
        else:
            msg = f"N1-Real-ESRGAN-x{scale}【待补】，使用 cv2.INTER_CUBIC 上采样"
        self._add_note(opts, msg)

    def _try_bind_realesrgan(self, weight: Path, scale: int) -> bool:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except Exception as e:
            logger.info("Real-ESRGAN 运行时未安装（%s），保留按需加载钩子", e)
            return False
        try:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=scale,
            )
            self._upsampler = RealESRGANer(
                scale=scale,
                model_path=str(weight),
                model=model,
                tile=256,
                tile_pad=10,
                pre_pad=0,
                half=str(self.device).startswith("cuda"),
            )
            return True
        except Exception as e:
            logger.warning("Real-ESRGAN 按需加载失败: %s", e)
            self._upsampler = None
            return False

    def _try_bind_zero_dce(self, weight: Path) -> bool:
        try:
            import zero_dce  # type: ignore
        except Exception:
            return False
        loader = getattr(zero_dce, "load_model", None)
        if not callable(loader):
            logger.info("zero_dce 已安装但无 load_model，钩子不绑定")
            return False
        try:
            self._zero_dce = loader(str(weight), device=self.device)
            return self._zero_dce is not None
        except Exception as e:
            logger.warning("Zero-DCE++ 按需加载失败: %s", e)
            self._zero_dce = None
            return False

    def _try_bind_nafnet(self, weight: Path, kind: str) -> bool:
        try:
            import nafnet  # type: ignore
        except Exception:
            return False
        attr = "load_denoise" if kind == "denoise" else "load_deblur"
        loader = getattr(nafnet, attr, None)
        if not callable(loader):
            logger.info("nafnet 已安装但无 %s，钩子不绑定", attr)
            return False
        slot = "_naf_denoise" if kind == "denoise" else "_naf_deblur"
        try:
            model = loader(str(weight), device=self.device)
        except Exception as e:
            logger.warning("NAFNet %s 按需加载失败: %s", kind, e)
            return False
        setattr(self, slot, model)
        return model is not None

    def _stabilize_opencv(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            self._traj = np.zeros(3, dtype=np.float64)
            self._smooth = np.zeros(3, dtype=np.float64)
            return frame
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )
        if prev_pts is None or len(prev_pts) < 8:
            self._prev_gray = gray
            return frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
        if curr_pts is None or status is None:
            self._prev_gray = gray
            return frame
        ok = status.reshape(-1).astype(bool)
        prev_ok = prev_pts[ok]
        curr_ok = curr_pts[ok]
        if len(prev_ok) < 6:
            self._prev_gray = gray
            return frame
        matrix, _ = cv2.estimateAffinePartial2D(prev_ok, curr_ok)
        if matrix is None:
            self._prev_gray = gray
            return frame
        dx = float(np.clip(matrix[0, 2], -40, 40))
        dy = float(np.clip(matrix[1, 2], -40, 40))
        da = float(np.clip(np.arctan2(matrix[1, 0], matrix[0, 0]), -0.1, 0.1))
        assert self._traj is not None and self._smooth is not None
        self._traj = self._traj + np.array([dx, dy, da], dtype=np.float64)
        self._smooth = 0.85 * self._smooth + 0.15 * self._traj
        corr = self._smooth - self._traj
        c, s = float(np.cos(corr[2])), float(np.sin(corr[2]))
        warp = np.array([[c, -s, corr[0]], [s, c, corr[1]]], dtype=np.float32)
        h, w = frame.shape[:2]
        out = cv2.warpAffine(frame, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self._prev_gray = gray
        return out

    def _apply_lowlight(self, frame: np.ndarray) -> np.ndarray:
        if self.backend.get("lowlight") == "zero_dce" and self._zero_dce is not None:
            try:
                out = self._zero_dce(frame)
                if isinstance(out, np.ndarray):
                    return out
            except Exception as e:
                logger.warning("Zero-DCE++ 推理失败（%s）【待补】，改用 gamma/CLAHE", e)
        return self._gamma_clahe(frame)

    def _gamma_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        light, a, b = cv2.split(lab)
        h, w = light.shape[:2]
        if h >= 8 and w >= 8:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            light = self._clahe.apply(light)
        if self._gamma_lut is None:
            gamma = 0.75
            self._gamma_lut = np.array(
                [min(255, int(round((i / 255.0) ** gamma * 255.0))) for i in range(256)],
                dtype=np.uint8,
            )
        light = cv2.LUT(light, self._gamma_lut)
        return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)

    def _apply_denoise(self, frame: np.ndarray) -> np.ndarray:
        if self.backend.get("denoise") == "nafnet" and self._naf_denoise is not None:
            try:
                out = self._naf_denoise(frame)
                if isinstance(out, np.ndarray):
                    return out
            except Exception as e:
                logger.warning("NAFNet 去噪推理失败（%s）【待补】，改用 fastNlMeans", e)
        return cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 15)

    def _apply_deblur(self, frame: np.ndarray) -> np.ndarray:
        if self.backend.get("deblur") == "nafnet" and self._naf_deblur is not None:
            try:
                out = self._naf_deblur(frame)
                if isinstance(out, np.ndarray):
                    return out
            except Exception as e:
                logger.warning("NAFNet 去糊推理失败（%s）【待补】，改用反锐化掩模", e)
        blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
        return cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

    def _apply_superres(self, frame: np.ndarray, scale: int) -> np.ndarray:
        if self.backend.get("superres") == "realesrgan" and self._upsampler is not None:
            try:
                out, _ = self._upsampler.enhance(frame, outscale=scale)
                if isinstance(out, np.ndarray):
                    return out
            except Exception as e:
                logger.warning("Real-ESRGAN 推理失败（%s）【待补】，改用 INTER_CUBIC", e)
        h, w = frame.shape[:2]
        return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
