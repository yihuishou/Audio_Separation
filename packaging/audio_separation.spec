# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec（PLAN §2.13 依赖分组 / M7 打包发布）。

## 核心思路：只打包「程序」，重依赖交给目标系统

实测（本机 RX 7900 XTX / ROCm / torch 2.9.1+rocmsdk20260116）：

| 配置 | 产物体积 |
|------|----------|
| 本 spec（torch 走系统） | **~120 MB** |
| 全部打包（对照） | 1.1 GB |

torch / numpy / scipy / librosa **不打包**，由目标系统自己装。
原因不是省事，是**正确性**：

1. ROCm 的 ``rocsolver.dll``(109MB) / ``rocsparse.dll``(99MB) / ``rocblas.dll``(41MB)
   是 PyInstaller 顺着 torch 的二进制依赖链收进来的。目标系统 ROCm 版本若与
   构建机不同，bundle 里的旧 DLL 会抢先加载 → 版本冲突崩溃。
2. torch 是按**它自己那份** numpy 编译的，bundle 里换一份 numpy 会直接崩。

所以策略是：excludes 排掉这些包，运行时由 ``rthook_syspath.py`` 把目标系统的
site-packages 加回 ``sys.path``（**追加到末尾**，bundle 内模块优先）。

## 产物布局（用户手动放 models / ffmpeg）

    dist/audio_separation/
    ├── audio_separation.exe      ← 双击这个
    ├── _internal/                ← PyInstaller 内部目录
    ├── models/                   ← 用户手动复制（1.2 GB）
    └── ffmpeg/                   ← 用户手动复制（315 MB）

★ 注意 models/ffmpeg 放在 **exe 旁边**，不是 ``_internal/`` 里。
``paths.py`` 的第④路探测（从 ``sys.executable`` 向上）就是为这个准备的。

## 构建

    python -m PyInstaller packaging/audio_separation.spec --noconfirm

或用 ``tools/build_windows.ps1`` / ``tools/build_linux.sh``。
"""
from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# ── 路径 ────────────────────────────────────────────────────────────
SPECPATH = Path(SPECPATH).resolve()
REPO = SPECPATH.parent
SRC = REPO / "src"

# ── 排除：交给目标系统的包 ──────────────────────────────────────────
# torch 系（含 ROCm SDK）—— 目标系统自己装，版本必须匹配本机驱动
TORCH_EXCLUDES = [
    "torch",
    "torchaudio",
    "torchvision",
    "torchgen",
    "torchlibrosa",
    "torchsummary",
    "triton",
    "nvidia",
    "_rocm_sdk_core",
    "_rocm_sdk_libraries_custom",
    "_rocm_sdk_libraries_ocr",
    "_rocm_sdk_libraries_visual",
]

# 科学计算栈 —— 系统装 torch 时必然带上 numpy，librosa 依赖 scipy/numba
# ★ 排掉它们，运行时从系统取，避免版本错配
SCIENCE_EXCLUDES = [
    "numpy",
    "numpy.libs",
    "scipy",
    "scipy.libs",
    "librosa",
    "numba",
    "llvmlite",
    "soundfile",
    "soxr",
    "audioread",
    "pooch",
    "joblib",
    "threadpoolctl",
    "decorator",
    "lazy_loader",
    "msgpack",
]

# ★★BS-RoFormer 家族 —— 必须走目标系统，理由和 torch 一样，但更硬：
#
#   PoPE_pytorch/triton_pope.py 里有 @triton.jit 装饰的函数。triton 的
#   JITCallable.__init__（triton/runtime/jit.py:471）会调 inspect.getsourcelines(fn)
#   去读**源码**。而冻结后纯 Python 模块在 PYZ 里只有 .pyc，读不到源码 →
#   OSError: could not get source code → ValueError: @jit functions should be
#   defined in a Python file。
#
#   PoPE_pytorch/attention.py 里的 try/except 只接 ImportError，接不住 ValueError，
#   于是 import bs_roformer 整个失败 —— 表现为冻结态一启动就崩，跟 CUDA 无关。
#
#   所以这组包必须从**磁盘**加载（目标系统 pip install BS-RoFormer 一条命令全带上）。
#   顺带好处：bs_roformer 与目标系统的 torch 版本天然对齐，不会错配。
#
#   ★ 注意包名：PyPI 上是 pope-pytorch（小写），但 **import 名是大写的 PoPE_pytorch**。
#     小写那个在 PyInstaller 里查不到，会报 "Hidden import 'pope_pytorch' not found"。
BS_ROFORMER_EXCLUDES = [
    "bs_roformer",
    "PoPE_pytorch",
    "einops",
    "einx",
    "torch_einops_utils",
    "rotary_embedding_torch",
    "hyper_connections",
    "beartype",
    "frozendict",
]

# 明确用不到的大块头（多为 torch 的 optional extras，PyInstaller 会顺带收进来）
HEAVY_EXCLUDES = [
    "transformers", "timm", "onnx", "onnxruntime", "safetensors", "tokenizers",
    "huggingface_hub", "accelerate", "datasets", "tensorboard", "z3",
    "matplotlib", "pandas", "PIL", "cv2", "imageio", "av", "polars",
    "_polars_runtime_32", "pythainlp", "nlpo3", "cryptography", "sympy",
    "networkx", "mpmath", "IPython", "jupyter", "notebook", "pytest",
    "setuptools", "pkg_resources", "filelock", "fsspec", "jinja2",
    "requests", "urllib3", "certifi", "charset_normalizer", "idna", "click",
    "MarkupSafe", "zipp", "importlib_metadata", "jaraco", "platformdirs",
    "texttable", "tabulate", "colorama", "typer", "shellingham", "rich",
    "markdown", "markdown_it", "mdurl", "Pygments", "bleach", "nltk",
    "regex", "joblib", "absl", "grpc", "google", "protobuf", "flatbuffers",
    "werkzeug", "sklearn", "pydantic", "pydantic_core", "sqlalchemy",
    "alembic", "flask", "django", "sphinx", "docutils", "IPython",
]

# PySide6 的无用子模块（QtWebEngine 单独就 350MB）
QTSIDE_EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput", "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras", "PySide6.Qt3DPhysics",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
    "PySide6.QtQuickControls2", "PySide6.QtQml",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtSpatialAudio",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtXml", "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtDesigner", "PySide6.QtHelp",
    "PySide6.QtPdf", "PySide6.QtPrintSupport", "PySide6.QtUiTools",
    "PySide6.QtRemoteObjects", "PySide6.QtStateMachine", "PySide6.QtScxml",
    "PySide6.QtWebSockets", "PySide6.QtWebChannel", "PySide6.QtTextToSpeech",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtNetworkAuth",
    "PySide6.QtHttpServer", "PySide6.QtLocation",
]

# ── 隐藏导入：走系统的包需要的 stdlib ───────────────────────────────
# ★ 这份清单是 ast 扫过 site-packages 里 torch / sympy / mpmath / filelock /
#   fsspec / jinja2 / networkx / numpy / scipy / librosa / numba / llvmlite
#   等**全部走系统的包**的 .py 收集到的 import 语句得到的 145 个标准库。
#
# ★ 为什么必须扫得这么全：走系统的包，它 import 到的每一个 stdlib 模块
#   都必须存在于 bundle 里。少一个就断链，而且是隔着包断的 ——
#   实际踩到过：mpmath 需要 colorsys，colorsys 不在 bundle，
#   于是系统的 mpmath → sympy → torch 全线崩，报错却是
#   「SymPy now depends on mpmath」，完全指不到真凶。
#
#   只扫 torch+triton 是不够的：torch 自己不 import colorsys，
#   是它的依赖 mpmath import 的。
STDLIB_HIDDEN = [
    "__future__", "_codecs", "_collections", "_collections_abc", "_compat_pickle",
    "_ctypes", "_imp", "_multiprocessing", "_operator", "_posixsubprocess",
    "_socket", "_string", "_thread", "_warnings", "_winapi", "abc", "argparse",
    "array", "ast", "asyncio", "atexit", "base64", "binascii", "bisect", "builtins",
    "bz2", "cProfile", "cmath", "code", "codecs", "collections", "colorsys",
    "concurrent", "configparser", "contextlib", "contextvars", "copy", "copyreg",
    "csv", "ctypes", "dataclasses", "datetime", "decimal", "difflib", "dis",
    "doctest", "enum", "errno", "faulthandler", "fcntl", "fileinput", "fnmatch",
    "fractions", "ftplib", "functools", "gc", "getpass", "gettext", "glob", "gzip",
    "hashlib", "heapq", "html", "importlib", "inspect", "io", "ipaddress",
    "itertools", "json", "keyword", "linecache", "locale", "logging", "lzma",
    "marshal", "math", "mmap", "msvcrt", "multiprocessing", "numbers", "opcode",
    "operator", "optparse", "os", "pathlib", "pdb", "pickle", "pickletools",
    "pkgutil", "platform", "posix", "posixpath", "pprint", "pstats", "pydoc",
    "queue", "random", "re", "readline", "resource", "rlcompleter", "runpy",
    "secrets", "select", "selectors", "shlex", "shutil", "signal", "site",
    "socket", "sqlite3", "ssl", "stat", "statistics", "string", "struct",
    "subprocess", "sys", "sysconfig", "tarfile", "tempfile", "textwrap",
    "threading", "time", "timeit", "tokenize", "tomllib", "trace", "traceback",
    "tracemalloc", "types", "typing", "unicodedata", "unittest", "urllib", "uuid",
    "warnings", "wave", "weakref", "winreg", "xml", "zipfile", "zipimport", "zlib",
    "zoneinfo",
]

# ★ 光有顶层模块不够：ctypes.wintypes / xml.etree.ElementTree 这类**子模块**
# 是运行时按需 import 的，顶层 AST 扫描扫不到，必须显式收。
SUBMODULE_HIDDEN = (
    collect_submodules("ctypes")
    + collect_submodules("xml")
    + collect_submodules("importlib")
    + collect_submodules("json")
    + collect_submodules("logging")
    + collect_submodules("unittest")
    + collect_submodules("concurrent")
    + collect_submodules("asyncio")
    + collect_submodules("email")
    + collect_submodules("http")
    + collect_submodules("urllib")
    + collect_submodules("collections")
    + collect_submodules("multiprocessing")
    + collect_submodules("sqlite3")
    + collect_submodules("decimal")
    + collect_submodules("pydoc")
)

# 本项目自己的包 + 直接依赖
# ★ bs_roformer 及其依赖（einops / einx / beartype / hyper_connections /
#   rotary_embedding_torch / torch_einops_utils）**不显式列** ——
#   PyInstaller 扫 bs_roformer 的 .py 时会自动发现它们。
#   唯一需要手工补的是被 excludes 排掉的 torch 的依赖（见上面的 STDLIB_HIDDEN）。
#   （pope-pytorch 同理不列：pyproject 声明了它，但项目与 bs_roformer 都没 import。）
PROJECT_HIDDEN = [
    "audio_separation",
    "yaml",
    "tqdm",
]

HIDDEN = sorted(set(STDLIB_HIDDEN) | set(SUBMODULE_HIDDEN) | set(PROJECT_HIDDEN))

# ── GPU 运行库 DLL：excludes 只管 Python 模块，管不到 DLL ────────────
# 保险起见再按文件名剔一遍。留着它们不仅白占 300MB，
# 还会让目标系统加载到与本地驱动不匹配的旧版本。
GPU_DLL_NAMES = (
    "rocsolver", "rocsparse", "rocblas", "rocfft", "rocrand", "rocsolver",
    "amdhip64", "amdhip", "amd_comgr", "hipblas", "hipblaslt", "hipsparse",
    "hipsolver", "hiprtc", "miopen", "rocprof", "rocm_agent", "roctx",
    "nvcuda", "nvrtc", "nvjitlink", "nvtx", "cudart", "cudart64", "cublas",
    "cublasLt", "cusparse", "cusolver", "curand", "cufft", "cudnn", "cudnn_engines",
    "cudnn_engines_precompiled", "cudnn_graph", "cudnn_heuristic", "cudnn_ops",
    "cudnn_adv", "nccl", "cupy", "c10_cuda", "torch_cuda", "torch_hip",
    "cudnn_frontend",
)


def _is_gpu_dll(name: str) -> bool:
    """判断一个二进制文件是不是 GPU 运行库。

    ★ Linux 的命名规则和 Windows 完全不同，必须分开处理：
      Windows: rocsolver.dll
      Linux  : librocsolver.so.7.1.0   ← lib 前缀 + 带版本号的 .so
    只按 startswith 匹配前缀的话，Linux 上一个都命中不了，
    300MB ROCm 运行库会原样进包，目标系统加载到与本地驱动不匹配的旧版本。
    """
    stem = os.path.basename(name).lower()

    # 去掉版本号后缀：.so.7.1.0 → .so
    for sep in (".so.", ".dylib."):
        idx = stem.find(sep)
        if idx != -1:
            stem = stem[:idx]
            break

    for suffix in (".dll", ".so", ".dylib", ".pyd"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    # Linux 的共享库带 lib 前缀，去掉才能和上面的前缀表对上
    if stem.startswith("lib"):
        stem = stem[3:]

    return any(stem.startswith(prefix) for prefix in GPU_DLL_NAMES)


a = Analysis(
    [str(SPECPATH / "entry.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=[],
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(SPECPATH / "rthook_syspath.py")],
    excludes=(
        TORCH_EXCLUDES
        + SCIENCE_EXCLUDES
        + BS_ROFORMER_EXCLUDES
        + HEAVY_EXCLUDES
        + QTSIDE_EXCLUDES
    ),
    noarchive=False,
)

# ── 剔掉漏网的 GPU DLL ──────────────────────────────────────────────
_before = len(a.binaries)
a.binaries = TOC([x for x in a.binaries if not _is_gpu_dll(x[0])])
print(f"[spec] GPU DLL 剔除：{_before} -> {len(a.binaries)} 条二进制")

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="audio_separation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # ★ 必须 False（GUI 子系统）：双击 exe 不弹黑色控制台窗口。
    #
    #   实测（PyInstaller 6.22.3，Windows，onedir）：console=False 时，
    #   父进程用 stdout=PIPE 拉起的子进程，其 stdout 仍然是可用的管道
    #   —— worker 的 JSON Lines 协议收发完全正常。所以 GUI 模式无窗口
    #   与 worker 模式有管道并不矛盾。
    #
    #   剩下的边角由 src/audio_separation/core/console_hider.py 兜底
    #   （GUI 分支里再调一次 FreeConsole）。
    #
    # ★ Linux 上这个参数是**无效的**，不是没配对：
    #   PyInstaller 官方文档对 `--windowed` 的原文是
    #   「This option is ignored on *NIX systems.」
    #   所以 Linux 产物就是普通控制台程序 —— stdout 照常工作
    #   （worker 协议天然可用，不依赖 Windows 那套管道技巧），
    #   从文件管理器双击也不会像 Windows 那样弹黑框
    #   （Linux 本来就没有「控制台窗口」这个概念）。
    #   同一个值通吃两个平台，无需分平台判断。
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="audio_separation",
)
