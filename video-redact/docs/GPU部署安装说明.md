# GPU 部署安装说明（Seetacloud / AutoDL）

面向本项目在 **AutoDL / Seetacloud** 云 GPU 上的部署与冒烟验证。无容器；一卡一任务；模型按需加载。

| 项 | 值 |
|----|----|
| 工程目录（数据盘） | `/root/autodl-tmp/video-redact` |
| SSH 示例 | `ssh -p 11118 root@connect.westc.seetacloud.com` |
| 典型 GPU | RTX 5090 D（约 32GB 显存） |
| CUDA | 主机常见 **12.8**；驱动可能报更高（如 13.x），以 `nvidia-smi` 为准 |
| Python | Miniconda **3.12**（系统常预装） |
| Web 端口 | **6006**（AutoDL 常用映射口，可改） |
| 代码仓库 | GitHub `okgo` 仓库下的 `video-redact/` 子目录 |

> 密码请放在控制台或密钥登录；聊天中出现过的密码建议在面板里轮换。

---

## 1. SSH 登录

```bash
ssh -p 11118 root@connect.westc.seetacloud.com
# 或你控制台给出的 host / 端口
```

登录后建议先确认硬件：

```bash
nvidia-smi
nvcc --version 2>/dev/null || true
python --version
which python
df -h /
df -h /root/autodl-tmp
```

| 检查 | 期望 |
|------|------|
| `nvidia-smi` | 看到 GPU 名称与显存 |
| `python` | 3.10+（本机多为 3.12） |
| `/` | 系统盘较小（约数十 GB），**不要**把权重和视频堆在这里 |
| `/root/autodl-tmp` | 数据盘，工程与权重放这里 |

---

## 2. 磁盘布局（务必读）

| 路径 | 用途 |
|------|------|
| `/`（系统盘） | 系统、Miniconda、少量工具；空间紧 |
| `/root/autodl-tmp/` | **数据盘**：工程、`.venv`、权重、样片、任务输出 |
| `/root/autodl-tmp/video-redact/` | 本项目根目录 |
| `weights/` | YuNet / 车牌 YOLO / 增强权重 |
| `weights/enhance/` | N1–N4 专用权重（可空；缺则 OpenCV 降级） |
| `samples/` | 合成烟测视频 |
| `jobs/` | 网页与 CLI 输出 |

```bash
mkdir -p /root/autodl-tmp/video-redact/{weights/enhance,samples,jobs}
cd /root/autodl-tmp/video-redact
```

从 Git 拉取（博客仓库子目录）：

```bash
# 若整仓 clone：
git clone https://github.com/xiongwei-sudo/okgo.git /tmp/okgo
cp -a /tmp/okgo/video-redact/. /root/autodl-tmp/video-redact/

# 若已有 sparse / 手动上传源码包，解压到上述目录即可
```

启用 Git LFS（权重后续进库时需要）：

```bash
# 若系统无 git-lfs：
# apt-get update && apt-get install -y git-lfs
git lfs install
cd /root/autodl-tmp/video-redact   # 或完整 okgo 仓库根
git lfs pull   # 拉取已跟踪的 .pt / .onnx
```

`.gitattributes` 已配置：`weights/**/*.pt`、`weights/**/*.onnx` 走 LFS。

---

## 3. 环境安装（两种方式）

`requirements.txt` **不含** `torch` / `torchvision`，需按下面 A 或 B 处理。

### 方式 A（推荐）：复用云机预装 PyTorch

AutoDL 镜像常已带系统级 CUDA 版 PyTorch。只补项目依赖，省时间与磁盘：

```bash
cd /root/autodl-tmp/video-redact

# 确认预装
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

若上面 `avail` 为 `True`，再装其余依赖（**不再**装 torch）：

```bash
python -m pip install -U pip setuptools wheel
# 国内可加镜像加速，例如：
# export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
# export PIP_TRUSTED_HOST=mirrors.aliyun.com
python -m pip install -r requirements.txt
```

可选：用 `venv` 隔离，但 `--system-site-packages` 继承预装 torch：

```bash
cd /root/autodl-tmp/video-redact
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 方式 B：独立 venv + cu128 轮子（预装不可用或版本不对时）

适用于系统 torch 过旧、无 CUDA、或与 RTX 5090 / CUDA 12.8 不匹配：

```bash
cd /root/autodl-tmp/video-redact
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# 官方 cu128（体积大，国外源可能慢）
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

# 若官方过慢，可改用国内 PyPI 镜像先装其它包，再对 torch 单独指定 index；
# 也可尝试阿里云等镜像站上的同名轮子（以镜像说明为准）：
# python -m pip install torch torchvision -i https://mirrors.aliyun.com/pypi/simple/

python -m pip install -r requirements.txt
```

也可参考仓库内 `scripts/setup_env.sh`（会建 conda 前缀并尝试 cu128→cu124）。脚本默认路径为 `/root/autodl-tmp/video-redact`。

### 公共依赖与 ffmpeg

```bash
# requirements.txt 主要包含：
# opencv-python-headless, ultralytics, mediapipe, onnxruntime-gpu,
# rapidocr-onnxruntime, fastapi, uvicorn, python-multipart, ...

command -v ffmpeg >/dev/null || apt-get update && apt-get install -y ffmpeg
```

### 版本核对

```bash
cd /root/autodl-tmp/video-redact
source .venv/bin/activate 2>/dev/null || true   # 若用了 venv
python scripts/print_versions.py 2>/dev/null || python - <<'PY'
import torch, cv2
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("opencv", cv2.__version__)
PY
```

---

## 4. 下载与放置模型权重

### 4.1 一键脚本（人脸 YuNet + 尝试车牌）

```bash
cd /root/autodl-tmp/video-redact
source .venv/bin/activate 2>/dev/null || true
python scripts/download_weights.py
```

脚本行为：

| 目标 | 结果 |
|------|------|
| `weights/face_detection_yunet_2023mar.onnx` | 多源拉取（GitHub / jsDelivr / HF 镜像） |
| `weights/plate_yolov8n.pt`（或候选名） | 多源尽力下载；全失败则写 `weights/PLATE_WEIGHTS_PENDING.txt`（【待补】） |
| `weights/enhance/README.txt` | 占位说明；N1–N4 专用权重【待补】 |

人脸推理优先 **MediaPipe**（pip 自带）；失败才用 YuNet onnx。

OCR（RapidOCR）首次推理时自动拉 det/rec，无需事先放进 `weights/`。

### 4.2 增强权重目录（N1–N4）

放到 `weights/` 或 `weights/enhance/`，文件名能被通配匹配即可（见 `pipeline/enhance.py`）：

| 模块 | 建议文件名关键词 | 缺省行为 |
|------|------------------|----------|
| N1 超分 Real-ESRGAN | `*RealESRGAN*` / `*ESRGAN*` / `*x2*` | OpenCV 同分辨率增强 |
| N2 低光 Zero-DCE++ | `*Zero-DCE*` / `*dce*` | CLAHE |
| N2 去噪 NAFNet | `*NAFNet*` / `*denoise*` | OpenCV 去噪 |
| N2 去糊 | `*Deblur*` | 锐化 |
| N3 稳像 | `*vidstab*` / `*stabil*` | 光流稳像 |
| N4 视频超分 | 后续接入 | 暂降级 |

当前代码在「发现权重但未接线」时也会记【待补】并走 OpenCV 降级；**没有权重也能跑通打码主流程**。

### 4.3 车牌【待补】时手动放置

若脚本未能下到公开权重，自行将 Ultralytics 兼容的 `.pt` 放到：

```text
/root/autodl-tmp/video-redact/weights/plate_yolov8n.pt
```

然后删除 `PLATE_WEIGHTS_PENDING.txt`（可选）。未就绪时流水线会用颜色/轮廓启发式，精度有限。

### 4.4 权重与 Git LFS

- 本地/云机：`python scripts/download_weights.py` 写入 `weights/`
- 入库：确保已 `git lfs track`（仓库已带 `.gitattributes`），再 `git add weights/*.pt weights/*.onnx && git commit && git push`
- Clone 方：`git lfs install && git lfs pull`，否则可能只拿到指针文件

大体积权重勿用普通 git 直推。

---

## 5. 冒烟测试（CLI）

```bash
cd /root/autodl-tmp/video-redact
source .venv/bin/activate 2>/dev/null || true

# 合成短样片
python scripts/make_sample.py -o samples/synthetic_smoke.mp4

# GPU 打码
python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_smoke.mp4 -v

# 或一键（脚本默认激活 .venv；若用方式 A 无 .venv，请直接跑上面两行）
bash scripts/smoke_test.sh
```

常用参数：

```bash
# 仅人脸 + 马赛克 + 强制 CPU
python cli.py -i in.mp4 -o out.mp4 --no-plate --no-text --mode mosaic --cpu

# 先增强再打码（缺增强权重 → OpenCV 降级）
python cli.py -i in.mp4 -o out.mp4 --enhance --stabilize --lowlight --denoise --superres --scale 2
```

成功标志：生成非空 `jobs/out_smoke.mp4`，日志无致命 Traceback；人脸检测应有命中（合成样片为人脸方块时视检测器而定）。

---

## 6. 启动 FastAPI 网页

```bash
cd /root/autodl-tmp/video-redact
source .venv/bin/activate 2>/dev/null || true
uvicorn web.app:app --host 0.0.0.0 --port 6006
```

- AutoDL 控制台通常把 **6006** 映射到「自定义服务」链接；浏览器打开即可上传视频、看进度、下载结果。
- 队列 **concurrency=1**（一卡一任务）；模型按任务加载，可卸载。
- 上传落盘在 `jobs/`；勿把超大历史结果堆满数据盘。

后台常驻示例：

```bash
nohup uvicorn web.app:app --host 0.0.0.0 --port 6006 > /tmp/video-redact-web.log 2>&1 &
```

---

## 7. 模型与运行约定（摘要）

| 能力 | 实现 | 部署注意 |
|------|------|----------|
| 人脸 | MediaPipe → YuNet onnx | `download_weights.py` 拉 YuNet；MediaPipe 随 pip |
| 车牌 | YOLOv8（ultralytics） | 公开权重可能【待补】；可手工放 `.pt` |
| 敏感文字 | RapidOCR + 身份证/手机号正则 | 首次推理自拉模型 |
| 增强 N1–N4 | Real-ESRGAN / NAFNet / Zero-DCE++ / vidstab 等 | 权重放入 `weights/enhance/`；缺则 OpenCV |
| 调度 | 内存队列 | 1 GPU / 1 job；无 Docker |

默认建议链路：**可选增强 → 检测 → 跟踪 → 打码 → ffmpeg 合并音轨**。

---

## 8. 常见问题

### pip / 下载很慢

```bash
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com
# HF 权重可优先走 hf-mirror（download_weights.py 已含镜像 URL）
```

torch cu128 官方 wheel 可达数百 MB～数 GB，耐心等待或改用方式 A。

### `torch.cuda.is_available()` 为 False

- 查 `nvidia-smi` 是否正常、驱动与 CUDA 是否匹配  
- 方式 B 是否装成了 CPU 版 torch（重新按 cu128 index 安装）  
- 临时用 `--cpu` 跑通流水线，再修 GPU

### 车牌效果差 / `PLATE_WEIGHTS_PENDING.txt`

公开 URL 可能失效。手动放入兼容的 `plate_yolov8n.pt` 后重试；否则仅为启发式，不能当生产车牌检测。

### 增强无明显效果

专用权重【待补】时走 OpenCV 降级（稳像/CLAHE/去噪/锐化），观感弱于 Real-ESRGAN 等。把对应 `.pth`/`.pt` 放进 `weights/enhance/` 并等代码接线后生效。

### Git LFS 只下到几十字节指针

```bash
git lfs install
git lfs pull
# 确认文件大小：ls -lh weights/
```

### 系统盘满

把工程、venv、pip cache、权重都放在 `/root/autodl-tmp`；清理 `~/.cache/pip` 时注意别误删仍要用的 wheel。

### `smoke_test.sh` 报找不到 `.venv`

脚本默认 `source .venv/bin/activate`。若采用方式 A 且未建 venv，直接手动跑 `cli.py`，或建一个带 `--system-site-packages` 的 `.venv`。

### 端口访问不了

确认 `--host 0.0.0.0`、AutoDL「自定义服务」映射端口与 `6006` 一致，以及实例未关机。

---

## 9. 检查清单（上线前）

- [ ] `nvidia-smi` 正常，工程在 `/root/autodl-tmp/video-redact`
- [ ] 方式 A 或 B 装好依赖；`torch.cuda.is_available()==True`（或明确接受 CPU）
- [ ] `python scripts/download_weights.py` 至少拿到 YuNet（或 MediaPipe 可用）
- [ ] `cli.py` 冒烟产出 `jobs/out_smoke.mp4`
- [ ] `uvicorn ... --port 6006` 可上传下载
- [ ] （可选）增强权重放入 `weights/enhance/`；车牌 `.pt` 就位
- [ ] （可选）权重经 Git LFS 推回仓库，他人 `git lfs pull` 可复现

---

## 10. 相关文件

| 文件 | 说明 |
|------|------|
| `README.md` | 项目总览与用法 |
| `requirements.txt` | 除 torch 外的 pip 依赖 |
| `scripts/setup_env.sh` | 云机 conda/venv + cu128 安装参考 |
| `scripts/download_weights.py` | 人脸/车牌权重下载 |
| `scripts/smoke_test.sh` | CLI 冒烟 |
| `scripts/make_sample.py` | 合成测试视频 |
| `cli.py` | 命令行入口 |
| `web/app.py` | FastAPI 网页 |
| `.gitattributes` | 权重 LFS 规则 |

文档版本：与仓库同步（2026-11）。实例端口与镜像预装情况以你当前 AutoDL 控制台为准。
