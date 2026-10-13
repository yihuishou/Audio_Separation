# -*- coding: utf-8 -*-
"""依赖清单 —— 单一事实来源（Single Source of Truth）。

从 PLAN.md §2.13 手工提取，实测本机版本。
所有检查、安装提示、GUI 面板的依赖数据都从这里读取，不分散在各处硬编码。

组别含义（PLAN §2.13）：
  A — 直接依赖（参考脚本实际 import 的）
  B — 传递依赖（BS-RoFormer 声明的 Requires，import 时连带拉起）
  C — 仅 GUI 层需要
  D — 可选
  E — 开发/测试
  F — 项目内嵌资源，不是 pip 包
  G — ★只检测不安装（torch：后端由硬件决定，wheel 不在 PyPI 主源）
"""

# torch 的安装提示按后端分支（用户自己选，工具不替他装）。
# 存为模板字符串，L3 GUI 面板按探测结果展示对应的那一行。
# ★没有 cpu 分支：本工具只做 GPU 推理（PLAN §2.14），给出 CPU-only wheel 的
# 安装命令是一条把人引向死路的建议 —— 装出来的 torch 能通过 L1（包在、版本对）
# 却过不了 L2，用户只会得到一个更晚、更难懂的失败。
TORCH_INSTALL_HINTS = {
    "rocm": "pip install torch --index-url https://download.pytorch.org/whl/rocm",
    "cuda": "pip install torch --index-url https://download.pytorch.org/whl/cu124",
}

# 依赖条目定义。每条必须包含这五个键，缺一不可：
#   name          — 分发名（pip 包名），用于 importlib.metadata.version() 查找
#   group         — 组别标识（A-G）
#   min_version   — 最低可接受版本；L1 拿它和实际版本比对，低于则降级。
#                    ★F 组（内嵌资源）例外：捆绑二进制没有 pip 版本号
#   level         — 缺失时的严重级别：FAIL / WARN / INFO
#   install_hint  — 修复提示（可复制的命令），★纯字符串，任何代码都不得执行它
#
# 可选键：
#   is_python     — True 表示这是解释器本身，不查 pip 元数据
#   is_embedded   — True 表示项目内嵌资源（F 组），不是 pip 包
#   install_hints — 仅 torch 有：按后端分支的命令字典，L3 面板按探测结果选一行
#                   （torch 用复数键，因为「装哪个后端」取决于用户硬件，工具无权替他选）
REQUIREMENTS = [
    # Python 解释器本身（不是 pip 包，单独列出）
    {"name": "python", "group": "A", "min_version": "3.11", "level": "FAIL",
     "install_hint": "需要 Python 3.11 或更新版本", "is_python": True},

    # A 组 · 直接依赖
    {"name": "numpy", "group": "A", "min_version": "2.4.3", "level": "FAIL",
     "install_hint": "pip install numpy"},
    {"name": "PyYAML", "group": "A", "min_version": "6.0.3", "level": "FAIL",
     "install_hint": "pip install PyYAML"},
    {"name": "BS-RoFormer", "group": "A", "min_version": "1.2.4", "level": "FAIL",
     "install_hint": "pip install BS-RoFormer"},
    {"name": "scipy", "group": "A", "min_version": "1.16.1", "level": "FAIL",
     "install_hint": "pip install scipy"},

    # G 组 · ★只检测不安装（torch：后端由硬件决定）
    # ★min_version 不能填本机那个 "2.9.1+rocmsdk20260116"：那会把「ROCm 构建号」
    # 变成硬性下限，Linux 上装个 2.7.1+cu124 就被误判成版本不够。这里取
    # BS-RoFormer 1.2.4 自己声明的 torch>=2.0 作为能力下限，不做更严的约束。
    {"name": "torch", "group": "G", "min_version": "2.0", "level": "FAIL",
     "install_hint": "按硬件选后端安装（见 install_hints），本工具不代为安装",
     "install_hints": TORCH_INSTALL_HINTS, "is_detect_only": True},

    # B 组 · 传递依赖（BS-RoFormer 1.2.4 声明的 Requires）
    {"name": "beartype", "group": "B", "min_version": "0.22.9", "level": "FAIL",
     "install_hint": "pip install beartype"},
    {"name": "einops", "group": "B", "min_version": "0.8.1", "level": "FAIL",
     "install_hint": "pip install einops"},
    {"name": "einx", "group": "B", "min_version": "0.4.3", "level": "FAIL",
     "install_hint": "pip install einx"},
    {"name": "hyper-connections", "group": "B", "min_version": "0.4.11", "level": "FAIL",
     "install_hint": "pip install hyper-connections"},
    {"name": "librosa", "group": "B", "min_version": "1.0.0", "level": "FAIL",
     "install_hint": "pip install librosa（连带 numba + soundfile）"},
    {"name": "PoPE-pytorch", "group": "B", "min_version": "0.2.1", "level": "FAIL",
     "install_hint": "pip install PoPE-pytorch"},
    {"name": "rotary-embedding-torch", "group": "B", "min_version": "0.9.1", "level": "FAIL",
     "install_hint": "pip install rotary-embedding-torch"},
    {"name": "torch-einops-utils", "group": "B", "min_version": "0.1.36", "level": "FAIL",
     "install_hint": "pip install torch-einops-utils"},

    # C 组 · 仅 GUI 层需要
    {"name": "PySide6", "group": "C", "min_version": "6.12.0", "level": "WARN",
     "install_hint": "pip install PySide6"},
    {"name": "tqdm", "group": "C", "min_version": "4.67.1", "level": "WARN",
     "install_hint": "pip install tqdm"},

    # D 组 · 可选
    {"name": "torchaudio", "group": "D", "min_version": "2.0", "level": "INFO",
     "install_hint": "pip install torchaudio（可选）"},

    # E 组 · 开发/测试
    {"name": "pytest", "group": "E", "min_version": "8", "level": "WARN",
     "install_hint": "pip install pytest"},
    {"name": "pytest-timeout", "group": "E", "min_version": "2.3", "level": "WARN",
     "install_hint": "pip install pytest-timeout"},
    {"name": "pytest-cov", "group": "E", "min_version": "5", "level": "WARN",
     "install_hint": "pip install pytest-cov"},
    {"name": "imageio-ffmpeg", "group": "E", "min_version": "0.6.0", "level": "WARN",
     "install_hint": "pip install imageio-ffmpeg"},

    # F 组 · ★项目内嵌资源，不是 pip 包，不作为 pip 分布检查
    {"name": "ffmpeg", "group": "F", "level": "INFO", "is_embedded": True,
     "install_hint": "ffmpeg/ 目录随仓库分发，不需要 pip 安装"},
]
