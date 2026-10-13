# Audio Separation — 人声分离工具

一个跨平台(Windows + Linux)的图形界面工具,用于将人声与伴奏分离。基于 BS-RoFormer v2/v3 模型,模型文件放置于 `models/bs_roformer_revive/` 目录下。默认使用 v2 模型,可在界面中切换至 v3。

## 安装顺序(必须严格按此顺序)

1. **安装 Python ≥ 3.11**
   从 [python.org](https://python.org) 下载并安装。安装时勾选 "Add to PATH"。

2. **★ 按硬件手动安装 torch** (本工具只检测、不代为安装)
    - ROCm (AMD GPU): `pip install torch --index-url https://download.pytorch.org/whl/rocm`
    - CUDA (NVIDIA GPU): `pip install torch --index-url https://download.pytorch.org/whl/cu124`

    > **为什么这一步必须在第 3 步之前且必须手动?**
    > ROCm/CUDA 的 torch wheel 均位于 `download.pytorch.org`,不在 PyPI。
    > 同一个版本号对应完全不同的 wheel,依赖声明无法表达后端选择。
    > 在 AMD 机器上直接 `pip install torch` 会拉取 CUDA 构建并"成功"安装,
    > 导致装错后端且难以诊断。因此本工具只检测 torch 状态,给出对应后端的安装命令供复制。
    > 
    > **本工具只支持 GPU 推理**。没有 GPU 时会立即终止,不做 CPU 回退(PLAN §2.14)。

3. **安装项目依赖**
   ```bash
   pip install -e ".[dev]"
   ```
   这一步不安装 torch(见第 2 步说明),仅安装其余依赖。

4. **放置模型文件**
   将 v2 和/或 v3 模型的 ckpt + yaml 文件放入 `models/bs_roformer_revive/` 目录。

5. **ffmpeg**
   已在仓库内 `ffmpeg/bin/`。Linux 上需赋予执行权限: `chmod +x ffmpeg/bin/ffmpeg ffmpeg/bin/ffprobe`。

6. **启动工具**
   ```bash
   python tools/run_ui.py
   ```

## 测试

本工具使用 pytest 作为测试框架。测试套件位于 `tests/` 目录。

**快速测试集(默认,秒级,无需 GPU 和权重):**
```bash
pytest -q
```

**慢速测试集(推理级,60 秒切片端到端,分钟级):**
```bash
pytest -m slow
```

**打包测试集(需要 dist/ 存在):**
```bash
pytest -m packaging
```

**注意:** 测试音频(约 513 MB)已加入 `.gitignore`,需本地提供。测试使用的 60 秒切片由 `tools/make_test_clips.py` 从 `test_audio/` 中的原始素材切割生成。

## 已知问题

- **ROCm + Windows 路径空格**: 若 PyTorch 安装在含空格的路径(如 `C:\Program Files\Python...`),ROCm 的 CUDA 后端初始化可能失败。因此推理在一个独立的子进程中运行,以隔离此问题。
- 本轮不实现结果回放与波形显示(明确在范围外)。

## 打包发布

本工具支持通过 pip 安装为系统级命令,提供三个入口点:

| 入口点 | 说明 |
|--------|------|
| `audio-separation` | CLI 命令行入口(有控制台窗口) |
| `audio-separation-ui` | GUI 入口(有控制台窗口,适合调试) |
| `audio-separation-gui` | GUI 入口(Windows 无控制台窗口,适合最终用户) |

**构建并发布:**
```bash
pip install build twine
python -m build
twine upload dist/*
```

**安装:**
```bash
pip install audio-separation
```

**Linux 上安装 ROCm torch wheel:**
```bash
# 需要预先安装 ROCm SDK(如 rocm-libs,hip-runtime-amd 等系统包)
pip install torch --index-url https://download.pytorch.org/whl/rocm
```
