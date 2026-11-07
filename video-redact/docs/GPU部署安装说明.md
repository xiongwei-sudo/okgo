# GPU 部署安装说明（AutoDL / SeetaCloud）

把人脸检测、车牌检测和可选增强跑在 AutoDL（丝塔云 SeetaCloud）GPU 实例上。不使用容器。模型按任务按需加载，一卡同时只跑一个任务。

博客仓库在别的目录，本说明只覆盖子项目 `video-redact/`。

## 1. SSH 登录

在 AutoDL 控制台打开实例，复制 SSH 命令。端口不是固定的 22，以控制台显示为准。

```bash
ssh -p <端口> root@<区域主机>
```

登录后先看 GPU 和数据盘：

```bash
nvidia-smi
df -h /root/autodl-tmp
```

系统和代码都放数据盘 `/root/autodl-tmp`。系统盘较小，实例重置后容易丢，不要把权重和任务输出写到系统盘。

## 2. 目录

脚本默认工程根是 `/root/autodl-tmp/video-redact`（下面能直接看到 `cli.py`）。仓库本身是 Astro 博客，打码代码在子目录里。克隆后做一次链接，让默认路径和脚本一致：

```bash
git clone https://github.com/xiongwei-sudo/okgo.git /root/autodl-tmp/okgo
ln -sfn /root/autodl-tmp/okgo/video-redact /root/autodl-tmp/video-redact
cd /root/autodl-tmp/video-redact
mkdir -p weights/enhance samples jobs
```

| 路径 | 作用 |
|------|------|
| `/root/autodl-tmp/video-redact` | 代码、虚拟环境、权重、样片、任务输出 |
| `weights/face_detection_yunet_2023mar.onnx` | YuNet 人脸（MediaPipe 不可用时的后备） |
| `weights/plate_yolov8n.pt` | 车牌 YOLO。也可叫 `yolov8n-plate.pt` 或 `license_plate_yolov8n.pt` |
| `weights/enhance/` | Real-ESRGAN、NAFNet、Zero-DCE++、vidstab 等增强权重 |
| `samples/` | 合成烟测片。`*.mp4` 不进 git |
| `jobs/` | 网页上传和打码结果。不进 git |

## 3. 优先用镜像里的 PyTorch

AutoDL 镜像通常已经装好带 CUDA 的 PyTorch。先用它，不要先重装 torch。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

能打印 GPU 名称后，把其余依赖装进当前解释器。`requirements.txt` 不含 torch，不会把 GPU 版换成 CPU 版。

推荐再套一个带系统站点包的虚拟环境，这样 `scripts/smoke_test.sh` 里的 `.venv` 也能用，同时仍看得到镜像里的 torch：

```bash
cd /root/autodl-tmp/video-redact
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

确认 torch 还是原来的 GPU 版：

```bash
python scripts/print_versions.py
```

只有 `import torch` 失败，或 `torch.cuda.is_available()` 为假时，才另装轮子。RTX 50 系可先试 cu128，不行再 cu124：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 失败再试：
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

`scripts/setup_env.sh` 会新建 `/root/autodl-tmp/video-redact/.conda` 并安装 torch。镜像里已有可用 GPU torch 时不要跑它。

`onnxruntime-gpu` 若和镜像里已有的 onnxruntime 冲突，保留能 `import onnxruntime` 的那一份即可。RapidOCR 用它做检测和识别。

没有 ffmpeg 时补上（只用于把原视频音轨嵌回结果）：

```bash
apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
```

## 4. 人脸、车牌、OCR 权重

在工程根目录执行：

```bash
cd /root/autodl-tmp/video-redact
source .venv/bin/activate
python scripts/download_weights.py
```

脚本会做这几件事：

- **YuNet**：下载到 `weights/face_detection_yunet_2023mar.onnx`。来源依次是 OpenCV Zoo、jsDelivr、Hugging Face 镜像和 Hugging Face。
- **人脸运行时**：任务开始时先加载 MediaPipe（模型随 pip 包）。MediaPipe 导入或初始化失败才用 YuNet。两条路径都是按需加载，不是开机常驻。
- **车牌**：按公开地址尝试 YOLOv8n，成功则保存为 `weights/plate_yolov8n.pt`，并用 `--device cuda:0` 做检测。检测器只认 `plate_yolov8n.pt`、`yolov8n-plate.pt`、`license_plate_yolov8n.pt`。脚本若只下到 `plate_yolov5.pt`，当前流水线不会加载它，仍走蓝/绿牌颜色启发式。
- **车牌失败**：写入 `weights/PLATE_WEIGHTS_PENDING.txt`，日志记【待补】。打码继续，不中断。
- **OCR**：`download_weights.py` 不拉 OCR 权重。RapidOCR 在第一次识别身份证号或手机号时自行下载 det/rec。

人脸和车牌都在每个任务里加载，任务结束调用 `unload` 释放。网页队列 `max_workers=1`，不要同时开第二个 GPU 进程。

## 5. 增强模型放哪里

专用网络仍是【待补】。把文件放到下面任一位置即可被发现：`weights/` 或 `weights/enhance/`（推荐后者）。

| 模块 | 文件名需能匹配 | 缺权重或尚未接入时 |
|------|----------------|-------------------|
| N1 Real-ESRGAN | `*RealESRGAN*`、`*realesrgan*`、`*ESRGAN*`、`*x2*` / `*x4*` | 放大后再缩回原分辨率（`INTER_CUBIC` + 锐化） |
| N2 Zero-DCE++ 低光 | `*Zero-DCE*`、`*zero_dce*`、`*dce*` | CLAHE |
| N2 NAFNet 去噪 | `*NAFNet*`、`*nafnet*`、`*denoise*` | `fastNlMeans` |
| N2 NAFNet 去糊 | `*Deblur*`、`*deblur*` | 反锐化 |
| N3 vidstab | `*vidstab*`、`*stabil*`、`*.npz` | OpenCV 光流稳像 |

即使这些文件已经放好，当前 `pipeline/enhance.py` 也只记录「权重已发现但仍【待补】接入」，画面仍走 OpenCV。官方推理接上之前，不要指望换上 `.pth` 就启用 Real-ESRGAN / NAFNet / Zero-DCE++。

增强同样按需判断：只有命令行或网页勾选了对应开关才会走该步骤，任务结束卸掉稳像状态。

## 6. 命令行烟测

```bash
cd /root/autodl-tmp/video-redact
source .venv/bin/activate
python scripts/make_sample.py -o samples/synthetic_smoke.mp4
python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_smoke.mp4 -v --device cuda:0
```

GPU 初始化失败时改用 CPU，确认代码本身能跑通：

```bash
python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_cpu.mp4 -v --cpu --no-plate --no-text --mode mosaic
```

`.venv` 已按上一节建好时，也可以：

```bash
bash scripts/smoke_test.sh
```

带增强的一次试跑（无官方权重时日志里会出现【待补】，并使用 OpenCV）：

```bash
python cli.py -i samples/synthetic_smoke.mp4 -o jobs/out_enhance.mp4 -v \
  --enhance --stabilize --lowlight --denoise --superres --scale 2
```

## 7. 网页（端口 6006）

```bash
cd /root/autodl-tmp/video-redact
source .venv/bin/activate
uvicorn web.app:app --host 0.0.0.0 --port 6006
```

AutoDL 控制台「自定义服务」打开 **6006**。监听 `0.0.0.0`，这样平台代理能访问。页面和 API 同源，不需要 CORS。

上传会先写到该任务目录的临时文件，再改名为 `jobs/<id>/input.*`，然后进入单卡队列。页面上可以开关人脸、车牌、敏感文字和增强项。

## 8. Git LFS 与不要提交的大文件

`video-redact/.gitattributes` 已把下面路径标成 Git LFS：

- `weights/*.pt`、`weights/*.onnx`
- `weights/**/*.pt`、`weights/**/*.onnx`

在会提交权重的机器上：

```bash
cd /root/autodl-tmp/okgo
git lfs install
git lfs pull
```

远程上如果还没有这些对象，`git lfs pull` 不会变出 YuNet 或车牌文件，仍用 `python scripts/download_weights.py`。

样片、任务输出、`.venv`、`.conda`、`*.engine` 保持在数据盘，不要提交。权重若要入库，先确认 `git check-attr filter -- video-redact/weights/<文件>` 显示 `lfs`，再提交指针，不要把大二进制直接推进博客仓库。

## 9. 部署时记住的约束

- **无容器**：不要包 Docker 镜像，用实例上的 Python。
- **一卡一任务**：不要并行开多个 `cli.py` 或第二个 uvicorn worker。
- **按需加载**：人脸、车牌、OCR、增强都在任务内加载，结束卸载。
- **增强【待补】**：OpenCV 降级可以先上线；Real-ESRGAN / NAFNet / Zero-DCE++ / vidstab 的官方权重只是先放到 `weights/enhance/`。
