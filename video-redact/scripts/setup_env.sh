#!/usr/bin/env bash
# 在 AutoDL GPU 机上执行：创建 conda 前缀环境并安装依赖
set -euo pipefail
PROJ="${PROJ:-/root/autodl-tmp/video-redact}"
PREFIX="${PROJ}/.conda"
export PATH="/root/miniconda3/bin:$PATH"
source /root/miniconda3/etc/profile.d/conda.sh

mkdir -p "$PROJ"/{weights,samples,jobs,weights/enhance}
cd "$PROJ"

if [[ ! -d "$PREFIX" ]]; then
  conda create -y -p "$PREFIX" python=3.11
fi
conda activate "$PREFIX"

# 清华/AutoDL 常用：先装 torch cu128
python -m pip install -U pip setuptools wheel
# RTX 5090 (Blackwell) 优先 cu128；失败再试 cu124 / nightly
set +e
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
TORCH_OK=$?
if [[ $TORCH_OK -ne 0 ]]; then
  echo "cu128 wheel failed, trying cu124..."
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  TORCH_OK=$?
fi
if [[ $TORCH_OK -ne 0 ]]; then
  echo "official wheels failed, trying pip default torch..."
  python -m pip install torch torchvision
fi
set -e

python -m pip install -r requirements.txt

# ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
fi

python scripts/download_weights.py
python scripts/make_sample.py -o samples/synthetic_smoke.mp4

python - <<'PY'
import torch, cv2, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    try:
        x=torch.zeros(1,device="cuda"); print("cuda_tensor_ok", x.device)
    except Exception as e:
        print("cuda_tensor_FAIL", e)
print("opencv", cv2.__version__)
PY

echo "SETUP_DONE prefix=$PREFIX"
