N1 RealESRGAN_x4plus.pth — 官方 RRDBNet x4，pipeline 直接 torch 加载（无需 basicsr）
N2 zero_dce.pth — 官方 Zero-DCE（e_conv1–7），GPU 推理
N2 NAFNet — 【待补】OpenCV 去噪降级
N3 稳像 — OpenCV 光流（无独立大权重）
N4 BasicVSR++ — 【待补】
路径以任务 notes / UI enhance_path 为准：official_gpu:* 或 opencv_*_fallback
