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

## 免安装分发(exe / Linux 可执行程序)

除 pip 分发外,还能打包成**免安装**的可执行程序:解压即用,不需要 Python 环境。

### 设计:只打包「程序」,重依赖交给目标系统

产物体积 **约 120 MB**,而不是 1.1 GB。torch / numpy / scipy / librosa **不进包**,
由目标系统自己装。这样做不只是为了省空间,更是为了正确性:

- ROCm 的 `rocsolver.dll`(109MB)、`rocsparse.dll`(99MB)、`rocblas.dll`(41MB) 是
  打包器顺着 torch 的二进制依赖链收进来的。目标系统的 ROCm 版本若与构建机不同,
  这几个旧 DLL 会抢先加载 → 版本冲突崩溃。
- torch 是按**它自己那份** numpy 编译的,包里换一份 numpy 会直接崩。

所以运行时由 `packaging/rthook_syspath.py`(runtime hook)把目标系统的
site-packages 加回 `sys.path`,追加在**末尾** —— 包内模块优先,系统的只在包内没有时生效。

### 产物布局

```
dist/audio_separation/
├── audio_separation.exe      ← Windows 上双击这个
├── _internal/                ← 打包器内部目录(不要动)
├── models/                   ← 手动复制(约 1.2 GB)
└── ffmpeg/                   ← 手动复制(约 315 MB)
```

★ `models/` 和 `ffmpeg/` 放在 **exe 旁边**,不是 `_internal/` 里面。

### 目标系统需要准备什么

| 内容 | 原因 |
|------|------|
| ROCm 驱动(系统级) | 内核模块,无法随包分发。目标机器必须先装好对应发行版的 ROCm |
| `pip install torch` | 用与该机器驱动匹配的 ROCm/CUDA wheel |
| `pip install BS-RoFormer` | ★**必须**。见下方「为什么 BS-RoFormer 也不能打包」 |

**只有两条命令**,其余全部由 pip 自动带出:

```bash
# Linux(ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install BS-RoFormer==1.2.4

# Windows(ROCm 官方 Windows 预览版)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2.windows
pip install BS-RoFormer==1.2.4

# NVIDIA 显卡则用 cuXXX 对应的 index-url
```

> ★★**顺序不能反**。`BS-RoFormer` 声明了 `torch>=2.0`,若目标机器上还没装 torch,
> pip 会从 PyPI 拉**默认的 CUDA 构建** —— 在 AMD 机器上装出一个"装好了但用不了"的
> torch(约 2.5 GB),比干脆没装更难排查。
> 先用 `--index-url` 装好对应后端的 torch,pip 看到 `torch>=2.0` 已满足就不会再动它。

`pip install BS-RoFormer==1.2.4` 会自动带出下面这些(实测 pip 解析结果):

| 来源 | 包 |
|------|-----|
| BS-RoFormer 直接依赖 | `bs_roformer` `PoPE-pytorch` `beartype` `einops` `einx` `hyper-connections` `rotary-embedding-torch` `torch-einops-utils` |
| librosa 链 | `librosa` `numba` `llvmlite` `numpy` `scipy` `scikit-learn` `soundfile` `soxr` `pooch` `joblib` `decorator` `lazy-loader` `msgpack` `threadpoolctl` |
| einx 链 | `sympy` `mpmath` `frozendict` |
| torch 链 | `filelock` `typing-extensions` `setuptools` `networkx` `jinja2` `MarkupSafe` `fsspec` |
| 零碎 | `requests` `urllib3` `certifi` `idna` `charset-normalizer` `platformdirs` `cffi` `pycparser` `narwhals` `packaging` `cloudpickle` |

实测占盘 **约 400 MB**(不含 torch 本身)。

> `numpy` / `scipy` / `librosa` 也在这一链里,不用单独装;
> `PyYAML` / `tqdm` 则已经打进 exe,目标系统不需要装。


### 为什么 BS-RoFormer 也不能打包

BS-RoFormer 及其依赖(`bs_roformer` / `PoPE_pytorch` / `einops` / `einx` /
`beartype` / `hyper_connections` / `rotary_embedding_torch` / `torch_einops_utils`)
**全部不进包**,由目标系统装。理由和 torch 不同,更硬:

`PoPE_pytorch/triton_pope.py` 里有 `@triton.jit` 装饰的函数。triton 的
`JITCallable.__init__`(`triton/runtime/jit.py:471`)会调
`inspect.getsourcelines(fn)` 去读**源码**。而打包后纯 Python 模块在
`.pyc` 归档里,读不到源码:

```
OSError: could not get source code
→ ValueError: @jit functions should be defined in a Python file
```

`PoPE_pytorch/attention.py` 里的 `try/except` 只接 `ImportError`,
接不住 `ValueError`,于是 `import bs_roformer` 整个失败 —— 表现为
**冻结态一启动就崩,与 CUDA 无关**。

只有从**磁盘**加载(目标系统 pip 装的 `.py` 文件)才有源码给 triton 读。
顺带好处:bs_roformer 与目标系统的 torch 版本天然对齐,不会错配。

> ★ 命名陷阱:PyPI 上包名是 `pope-pytorch`(小写),但 **import 名是大写的
> `PoPE_pytorch`**。写 hidden import 时用小写会报
> `Hidden import 'pope_pytorch' not found`。

### 构建

```bash
# Windows
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1

# Linux(★ PyInstaller 不能跨平台编译,必须在 Linux 上构建)
bash tools/build_linux.sh
```

构建完按提示手动复制 `models/` 和 `ffmpeg/` 到 exe 旁边,即可分发。

### 四种运行模式

同一个 exe 有四种启动方式:

| 命令 | 行为 |
|------|------|
| `audio_separation` | 弹 GUI 窗口 |
| `audio_separation --version` | 打印版本号后退出(退出码 0) |
| `audio_separation --worker --job job.json` | worker 模式,stdout 吐 JSON Lines |
| `audio_separation --probe` | 探测 GPU(不弹窗口),结果写入 `PROBE_RESULT_FILE` 指定的文件 |

★ GUI 模式**不弹控制台窗口** —— 打包时用的是 `console=False`(GUI 子系统)。
worker 模式靠 stdout 传 JSON Lines 协议,实测父进程用 `stdout=PIPE` 拉起
无控制台的子进程仍能完整收到协议内容,隐藏窗口不会打断通信。

### 运行日志

GUI 模式会在 **exe 同级目录的 `logs/`** 下写日志,文件名 `app-<日期>.log`
(开发态则写在仓库根的 `logs/`)。记录内容包括:

- 程序里所有 `print()` 的内容
- `logging` 的输出
- GUI 日志面板里显示的每一行(worker 的输出、完成/失败/取消提示)

单个文件超过 10 MB 会轮转成 `app-<日期>.1.log`,旧文件保留 14 天。

★ 隐藏控制台窗口后,出问题时用户看不到任何输出 —— 日志文件是唯一的排查线索,
反馈问题时请一并提供 `logs/` 下的文件。

### Linux 注意事项

- `models/` `ffmpeg/` 的可执行权限位可能丢失(PyInstaller 官方 issue #3926
  承认不保留文件属性),运行时会自动 `chmod` 兜底,也可手动 `chmod +x`。
- `ffmpeg/` 必须是 **Linux 版**二进制(文件名无 `.exe` 后缀),Windows 版跑不了。
- glibc 版本要求 2.28–2.39,更老的系统跑不了 ROCm torch。
