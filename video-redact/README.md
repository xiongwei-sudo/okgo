# 视频打码工具（video-redact）

本地 GPU 视频脱敏：人脸 / 车牌 / 敏感文字检测后高斯或马赛克打码。增强在 Real-ESRGAN / NAFNet / Zero-DCE++ / vidstab 权重缺失时走 OpenCV 降级，并在日志里标明【待补】。

## 当前状态

- 网页：上传先写入 `jobs/<id>/input.*`，再提交单卡队列；页面有增强勾选和进度条。同源使用，不开启 CORS。
- 增强：稳像（vidstab 或 OpenCV 特征点）、低光（gamma/CLAHE）、去噪（fastNlMeans）、去糊（反锐化）、超分（`cv2.INTER_CUBIC`）。官方权重仍【待补】，加载钩子按需保留。
- 车牌：`scripts/download_weights.py` 会试公开 YOLOv8 车牌权重；失败时看 `weights/PLATE_WEIGHTS_PENDING.txt`，颜色/轮廓启发式仍在。
- Git LFS：`weights/*.pt` 与 `weights/*.onnx` 已写入 `.gitattributes`。这些文件仍被 `.gitignore` 排除，本仓库不提交大权重。工作区没有现成的小型 YuNet onnx。

## 约定

- **无容器**；单机 Python / conda
- **一卡一任务**（内存队列 concurrency=1）
- **按需加载模型**，任务结束可卸载
- **默认建议**：先增强、后打码（`--enhance` 与打码同 CLI；官方权重【待补】时用 OpenCV，不再原样透传）
- OCR **保留**（RapidOCR + 身份证/手机号正则）

## 目录

| 路径 | 说明 |
|------|------|
| `video-redact/` | 本子项目（与上层 Astro 博客分开） |
| `weights/` | YuNet / 车牌 YOLO / enhance 占位；大文件不入库 |
| `weights/PLATE_WEIGHTS_PENDING.txt` | 车牌权重说明与下载地址 |
| `pipeline/` | 检测、跟踪、打码、增强降级、队列 |
| `cli.py` | 命令行入口 |
| `web/app.py` | FastAPI（上传、进度、下载） |
| `samples/` | 合成烟测样片（媒体文件被 gitignore） |

## 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/video-redact/.conda
# 或: source /root/autodl-tmp/video-redact/.venv/bin/activate
cd /root/autodl-tmp/video-redact
```

可选：`pip install vidstab realesrgan basicsr`。未安装时对应步骤走 OpenCV，并记录【待补】。

## 用法

```bash
# 合成样片
python scripts/make_sample.py -o samples/synthetic_smoke.mp4

# 打码（GPU）
python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_smoke.mp4 -v

# 仅人脸 + 马赛克 + CPU
python cli.py -i in.mp4 -o out.mp4 --no-plate --no-text --mode mosaic --cpu

# 增强：无 Real-ESRGAN 权重时用三次插值上采样，日志含【待补】
python cli.py -i in.mp4 -o out.mp4 --enhance --superres --scale 2

# 低光 / 去噪 / 稳像 / 去糊同样可单开
python cli.py -i in.mp4 -o out.mp4 --lowlight --denoise --stabilize --deblur --cpu

# 网页（可选，不要额外加 CORS）
uvicorn web.app:app --host 0.0.0.0 --port 6006
```

## 模型

| 模块 | 状态 |
|------|------|
| 人脸 MediaPipe / YuNet | 安装后可用；YuNet onnx 由 `scripts/download_weights.py` 拉取，不提交进 git |
| 车牌 YOLOv8n | 公开权重多源尝试；失败则启发式。说明见 `weights/PLATE_WEIGHTS_PENDING.txt` 【待补】 |
| RapidOCR | pip 包；首次推理拉 det/rec |
| Real-ESRGAN / NAFNet / Zero-DCE++ / vidstab | 权重【待补】。无权重时：超分 `INTER_CUBIC`、去噪 `fastNlMeans`、低光 gamma/CLAHE、稳像 vidstab 或 OpenCV |

### PLATE_WEIGHTS_PENDING

检测器只认这几个文件名（放在 `weights/`）：

- `plate_yolov8n.pt`
- `yolov8n-plate.pt`
- `license_plate_yolov8n.pt`

`python scripts/download_weights.py` 会依次请求 Koushim、joker5914、OpenPathAI 的公开 YOLOv8 车牌权重，以及 Hugging Face 镜像。全部失败时刷新 `weights/PLATE_WEIGHTS_PENDING.txt`，流水线继续用启发式。

## Git LFS

`.gitattributes` 已跟踪：

- `weights/*.pt`
- `weights/*.onnx`
- `weights/**/*.pt`
- `weights/**/*.onnx`

`.gitignore` 仍忽略这些后缀以及 `weights/` 里的实际文件，只保留 `.gitkeep` 和 `PLATE_WEIGHTS_PENDING.txt`。需要本地权重时再下载，不要把大的 `.pt` / `.onnx` / `.mp4` 提交到这个博客仓库。

## 分期

- M1：CLI 人脸打码可跑通（本目录）
- M2：车牌 + 马赛克 + ffmpeg 音轨
- M3：OCR 敏感过滤 + 简易跟踪
- M4：队列 / 按需加载；网页先落盘再入队
- N1–N4：官方权重【待补】；OpenCV 降级已接入
