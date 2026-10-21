### 解压后还需要自己准备两样东西

1. **模型权重**（约 1.2 GB）—— 放到程序目录下的 `models/` 里
2. **ffmpeg**（约 330 MB）—— 放到程序目录下的 `ffmpeg/` 里

这两样都没进包：模型太大，ffmpeg 每个平台要单独下。

### 目标系统还要装的东西

程序不含 torch 和 BS-RoFormer，运行时从系统里找。AMD 显卡装 ROCm 版 torch：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install BS-RoFormer>=1.2
```

完整说明见仓库 README。
