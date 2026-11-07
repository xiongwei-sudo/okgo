# 视频打码工具（video-redact）

本地 GPU 视频脱敏：人脸 / 车牌 / 敏感文字检测后高斯或马赛克打码；可选增强（N1–N4）。

## 文档

| 文档 | 说明 |
|------|------|
| [docs/技术方案.md](docs/技术方案.md) | 总体方案：范围、架构、检测／打码／跟踪、GPU 调度、增强 N1–N4、分期 M1–M5、NFR |
| [docs/设计方案思路.md](docs/设计方案思路.md) | 已定决策：无容器、按需加载、一卡一任务、先增强后打码、模型清单、AutoDL 5090D、分期与未完成项 |
| [docs/GPU部署安装说明.md](docs/GPU部署安装说明.md) | 云 GPU 部署：SSH、磁盘、预装 torch vs cu128 venv、权重、冒烟、FastAPI:6006、常见问题 |
| [docs/README.md](docs/README.md) | 文档索引 |

## 部署（GPU / AutoDL）

完整步骤见：**[docs/GPU部署安装说明.md](docs/GPU部署安装说明.md)**。技术方案见 **[docs/技术方案.md](docs/技术方案.md)**。已定决策见 **[docs/设计方案思路.md](docs/设计方案思路.md)**。

## 状态（2026-11-07）

| 模块 | 状态 |
|------|------|
| CLI 打码流水线 | 已实现（检测→跟踪→打码→ffmpeg 音轨） |
| FastAPI 网页 | 已实现上传/进度/下载；表单含增强开关；上传原子落盘 |
| 人脸 MediaPipe / YuNet | 可用；YuNet onnx 由 `scripts/download_weights.py` 拉取 |
| 车牌 YOLO | 多源公开 URL 尝试；失败则启发式 + `weights/PLATE_WEIGHTS_PENDING.txt` 【待补】 |
| RapidOCR | pip；首次推理自拉 det/rec |
| 增强 N1–N4 | 专用权重【待补】；缺权重时 **OpenCV 降级**（稳像/CLAHE/去噪/锐化/同分辨率超分增强） |
| 权重 Git LFS | `.gitattributes` 已配置 `weights/**/*.pt|*.onnx` |

## 约定

- **无容器**；单机 Python / conda
- **一卡一任务**（内存队列 concurrency=1）
- **按需加载模型**，任务结束可卸载
- **默认建议**：先增强、后打码（`--enhance`；权重【待补】时 OpenCV 降级）
- OCR **保留**（RapidOCR + 身份证/手机号正则）

## 目录

| 路径 | 说明 |
|------|------|
| `/root/autodl-tmp/video-redact` | 云机工程与权重（数据盘） |
| `weights/` | YuNet / 车牌 YOLO / enhance 占位（LFS） |
| `pipeline/` | 检测、跟踪、打码、增强、队列 |
| `cli.py` | 命令行入口 |
| `web/app.py` | FastAPI 网页 |
| `samples/` | 合成烟测样片 |

## 环境

```bash
source /root/autodl-tmp/video-redact/.venv/bin/activate
cd /root/autodl-tmp/video-redact
python scripts/download_weights.py
bash scripts/smoke_test.sh
```

## 用法

```bash
# 合成样片
python scripts/make_sample.py -o samples/synthetic_smoke.mp4

# 打码（GPU）
python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_smoke.mp4 -v

# 仅人脸 + 马赛克 + CPU
python cli.py -i in.mp4 -o out.mp4 --no-plate --no-text --mode mosaic --cpu

# 增强（缺权重 → OpenCV 降级）
python cli.py -i in.mp4 -o out.mp4 --enhance --stabilize --lowlight --denoise --superres --scale 2

# 网页
uvicorn web.app:app --host 0.0.0.0 --port 6006
```

## 分期

- M1：CLI 人脸打码可跑通
- M2：车牌 + 马赛克 + ffmpeg 音轨
- M3：OCR 敏感过滤 + 简易跟踪
- M4：队列 / 按需加载 / 网页增强开关（进行中）
- N1–N4：专用权重实装；当前 OpenCV 降级可用
