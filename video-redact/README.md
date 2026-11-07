# 视频打码工具（video-redact）

本地 GPU 视频脱敏：人脸 / 车牌 / 敏感文字检测后高斯或马赛克打码；可选增强（N1–N4，一期 stub）。

## 约定

- **无容器**；单机 Python / conda
- **一卡一任务**（内存队列 concurrency=1）
- **按需加载模型**，任务结束可卸载
- **默认建议**：先增强、后打码（`--enhance` 与打码同 CLI；增强权重【待补】时透传）
- OCR **保留**（RapidOCR + 身份证/手机号正则）

## 目录

| 路径 | 说明 |
|------|------|
| `/root/autodl-tmp/video-redact` | 工程与权重（数据盘） |
| `weights/` | YuNet / 车牌 YOLO / enhance 占位 |
| `pipeline/` | 检测、跟踪、打码、增强 stub、队列 |
| `cli.py` | 命令行入口 |
| `web/app.py` | FastAPI 骨架（可选） |
| `samples/` | 合成烟测样片 |

## 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/video-redact/.conda
# 或: source /root/autodl-tmp/video-redact/.venv/bin/activate
cd /root/autodl-tmp/video-redact
```

## 用法

```bash
# 合成样片
python scripts/make_sample.py -o samples/synthetic_smoke.mp4

# 打码（GPU）
python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_smoke.mp4 -v

# 仅人脸 + 马赛克 + CPU
python cli.py -i in.mp4 -o out.mp4 --no-plate --no-text --mode mosaic --cpu

# 增强开关（权重【待补】时为透传 stub）
python cli.py -i in.mp4 -o out.mp4 --enhance --superres --scale 2

# 网页（可选）
uvicorn web.app:app --host 0.0.0.0 --port 6006
```

## 模型

| 模块 | 状态 |
|------|------|
| 人脸 MediaPipe / YuNet | 安装后可用；YuNet onnx 由 `scripts/download_weights.py` 拉取 |
| 车牌 YOLOv8n | 公开权重多源尝试；失败则启发式，见 `weights/PLATE_WEIGHTS_PENDING.txt` 【待补】 |
| RapidOCR | pip 包；首次推理拉 det/rec |
| Real-ESRGAN / NAFNet / Zero-DCE++ / vidstab | N1–N4 【待补】 |

## 分期

- M1：CLI 人脸打码可跑通（本目录）
- M2：车牌 + 马赛克 + ffmpeg 音轨
- M3：OCR 敏感过滤 + 简易跟踪
- M4：队列 / 按需加载 / 显存指标
- N1–N4：增强实装
