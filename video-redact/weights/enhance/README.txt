增强权重状态（2026-11-07）

- RealESRGAN_x4plus.pth、zero_dce.pth：可由下载脚本获取，GPU 上已在使用。
  二进制不进本仓库，以后走 Git LFS。
- NAFNet-SIDD、BasicVSR++：仍【待补】。OpenCV 降级保持生效。
- vidstab：没有大权重文件。稳像使用 OpenCV 光流。

缺专用权重或推理未接入时，pipeline/enhance.py 记录【待补】并走 OpenCV 降级。
