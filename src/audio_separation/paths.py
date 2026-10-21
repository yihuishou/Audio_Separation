# -*- coding: utf-8 -*-
"""★ 全仓库唯一的「路径真源」（M0-A 阶段）。

## 为什么需要它

迁移前参考脚本 `tools/bs_roformer_revive_separate.py` 硬编码 `REPO = Path(__file__).parent.parent`，
并依赖 `audio_ai.paths` 包（在本项目不存在）。本模块提供独立的路径解析能力。

## ★ 三条设计约束（从 audio_ai 继承）

**① `repo_root()` 只有「显式配置错误」才抛异常。**
- `AUDIO_SEPARATION_ROOT` 设了但指向的目录缺标记 → 抛 `ResourceNotFound`
- 自动探测全未命中 → 返回 best-effort 猜测路径，失败记进诊断

**② 判定用「双标记同时存在」，不是单标记。**
`models/` 且 `ffmpeg/` 同时存在，才认作仓库根。

**③ 错误信息必须列出「试过哪些目录、每个缺什么」。**
`diagnose()` 返回完整探测链。

## 解析优先级

```
① AUDIO_SEPARATION_ROOT 环境变量（显式覆盖）
② 从本文件向上最多 6 层
③ 从 cwd 向上最多 6 层
④ 从可执行文件向上最多 6 层（★冻结态必需：双击 exe 时 cwd 不在 exe 旁）
⑤ 都没有 → best-effort（返回猜测路径，记录诊断）
```
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# ★ 仓库根的判定标记：必须同时存在（见模块 docstring 约束②）
# ffmpeg/ 目录名跨平台统一，各平台带各自的二进制
ROOT_MARKERS: tuple[str, ...] = ("models", "ffmpeg")

ENV_ROOT = "AUDIO_SEPARATION_ROOT"
ENV_OUTPUT = "AUDIO_SEPARATION_OUTPUT"

#: 向上探测的最大层数
_MAX_UP = 6

_CACHE: dict[str, object] = {}


class ResourceNotFound(FileNotFoundError):
    """资源定位失败。★消息里必带 `diagnose()` 的完整探测链。"""


# ══════════════════════════════════════════════════════════════════════════════
# 探测
# ══════════════════════════════════════════════════════════════════════════════
def _missing(p: Path) -> list[str]:
    """`p` 相对仓库根还缺哪些标记（空列表 = 是根）。"""
    return [m for m in ROOT_MARKERS if not (p / m).exists()]


def _probe(start: Path) -> tuple[Path | None, list[str]]:
    """从 `start` 向上找仓库根。返回 (根 or None, 探测日志)。"""
    log: list[str] = []
    try:
        cur = start.resolve()
    except OSError:  # 极端：路径不可解析
        return None, [f"  {start}  ← 无法 resolve()"]
    if cur.is_file():
        cur = cur.parent
    for _ in range(_MAX_UP):
        miss = _missing(cur)
        if not miss:
            log.append(f"  ✔ {cur}")
            return cur, log
        log.append(f"  ✗ {cur}   缺 {', '.join(miss)}")
        if cur.parent == cur:  # 到达盘符根
            break
        cur = cur.parent
    return None, log


def _discover() -> tuple[Path, str]:
    """跑完整探测链。返回 (best-effort 根, 诊断文本)。"""
    parts: list[str] = []
    root: Path | None = None

    # ── ① 环境变量（显式覆盖）──────────────────────────────────────────
    env = os.environ.get(ENV_ROOT)
    if env:
        p = Path(env).resolve()
        miss = _missing(p)
        if miss:
            raise ResourceNotFound(
                f"{ENV_ROOT}={env!r} 指向的目录缺少 {', '.join(miss)}。\n"
                f"该目录内容：{sorted(x.name for x in p.iterdir())[:20] if p.is_dir() else '(不是目录)'}"
            )
        root = p
        parts.append(f"① {ENV_ROOT}={env}  ← 显式覆盖，命中")
    else:
        parts.append(f"① {ENV_ROOT} 未设置")

    # ── ② 从本文件向上 ────────────────────────────────────────────────
    r, log = _probe(Path(__file__))
    parts.append(f"② 从 {Path(__file__).parent} 向上：\n" + "\n".join(log))
    if r is not None and root is None:
        root = r

    # ── ③ 从 cwd 向上 ─────────────────────────────────────────────────
    r, log = _probe(Path.cwd())
    parts.append(f"③ 从 cwd={Path.cwd()} 向上：\n" + "\n".join(log))
    if r is not None and root is None:
        root = r

    # ── ④ 从可执行文件向上（冻结态必需）──────────────────────────────
    # 双击 exe 时 cwd 不在 exe 旁（可能是 C:\Windows\System32），③ 必然落空。
    # exe 所在目录才是资源该在的地方（PLAN R10）。
    exe = self_executable()
    r, log = _probe(exe)
    parts.append(f"④ 从可执行文件 {exe} 向上：\n" + "\n".join(log))
    if r is not None and root is None:
        root = r

    if root is not None:
        parts.append(f"★ 解析成功：{root}")
    else:
        # best-effort：从 src/audio_separation/paths.py 上溯 3 层到仓库根
        guess = Path(__file__).resolve().parents[3]
        parts.append(
            f"★ 全部未命中。best-effort 猜 {guess}\n"
            f"   ⇒ 纯路径拼接仍可用，但 require() 会失败。\n"
            f"   ⇒ 修法：设 {ENV_ROOT} 环境变量指向含 "
            f"{' 与 '.join(ROOT_MARKERS)} 的目录。")
        root = guess
    return root, "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# 公开 API
# ══════════════════════════════════════════════════════════════════════════════
def repo_root() -> Path:
    """尽力而为地定位仓库根。★**不抛异常**（见 docstring 约束①）。"""
    if "root" not in _CACHE:
        root, diag = _discover()
        _CACHE["root"] = root
        _CACHE["diag"] = diag
    return _CACHE["root"]  # type: ignore[return-value]


def diagnose() -> str:
    """人类可读的完整探测链（`--help` / 环境检查会用）。"""
    repo_root()
    return str(_CACHE.get("diag", ""))


def models_dir() -> Path:
    """模型根目录（`models/`）。★纯拼接，不检查存在性。"""
    return repo_root() / "models"


def ffmpeg_dir() -> Path:
    """内置 ffmpeg 的 bin 目录（`ffmpeg/bin/`）。"""
    return repo_root() / "ffmpeg" / "bin"


def ffmpeg_bin() -> Path:
    """内置 ffmpeg 可执行文件。★**固定路径**，按平台选择可执行文件名。
    
    Windows: ffmpeg/bin/ffmpeg.exe
    Linux:   ffmpeg/bin/ffmpeg
    """
    if platform.system() == "Windows":
        return ffmpeg_dir() / "ffmpeg.exe"
    else:
        return ffmpeg_dir() / "ffmpeg"


def ffprobe_bin() -> Path:
    """内置 ffprobe 可执行文件。按平台选择可执行文件名。"""
    if platform.system() == "Windows":
        return ffmpeg_dir() / "ffprobe.exe"
    else:
        return ffmpeg_dir() / "ffprobe"


def find_ff(name: str | None = None) -> Path | None:
    """找 ffmpeg 系列的任一可执行文件。找不到返回 None（**不抛**）。
    
    如果未指定 name，则自动按平台选择 ffmpeg。
    顺序：内置目录 → PATH → 仓库内 rglob 兜底。
    """
    if name is None:
        # 自动选择平台对应的 ffmpeg 可执行文件名
        name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    
    cand = ffmpeg_dir() / name
    if cand.exists():
        return cand
    found = shutil.which(name)
    if found:
        return Path(found)
    hits = sorted(repo_root().rglob(name))
    return hits[0] if hits else None


def output_dir() -> Path:
    """产物目录（默认 `output/separated/`）。可被 `AUDIO_SEPARATION_OUTPUT` 覆盖。
    
    ★**不 mkdir** —— 纯路径函数不该有副作用；需要写入的调用点自己建。
    """
    env = os.environ.get(ENV_OUTPUT)
    if env:
        return Path(env)
    return repo_root() / "output" / "separated"


def test_audio_dir() -> Path:
    """测试素材根（`test_audio/`）。"""
    return repo_root() / "test_audio"


def test_clips_dir() -> Path:
    """测试切片目录（`test_audio/clips/`）。"""
    return test_audio_dir() / "clips"


def self_executable() -> Path:
    """返回当前 Python 可执行文件路径。
    
    在冻结构建（PyInstaller 等）中，返回 exe 本身。
    """
    # PyInstaller 打包后 sys.executable 指向 exe
    return Path(sys.executable)


def logs_dir() -> Path:
    """日志目录：exe 同级目录下的 `logs`。

    冻结态取 exe 所在目录 —— 用户把 exe 放哪，日志就跟着放哪，
    不用去猜安装路径。开发态取仓库根。
    """
    if getattr(sys, "frozen", False):
        return self_executable().resolve().parent / "logs"
    return repo_root() / "logs"


def bs_roformer_revive_dir() -> Path:
    """BS-RoFormer-Revive 模型目录（`models/bs_roformer_revive/`）。"""
    return models_dir() / "bs_roformer_revive"


def require(p: Path | str, what: str) -> Path:
    """确认 `p` 存在，否则抛带**完整探测链**的 `ResourceNotFound`。"""
    p = Path(p)
    if p.exists():
        return p
    raise ResourceNotFound(
        f"缺少{what}：{p}\n"
        f"★ 路径解析链：\n{diagnose()}\n"
        f"★ 若是资源没下载，检查 {models_dir()}/ 下该模型是否完整。"
    )


def require_dir(p: Path | str, what: str) -> Path:
    """同 `require()`，但要求是**目录**。"""
    p = require(p, what)
    if not p.is_dir():
        raise ResourceNotFound(f"{what} 不是目录：{p}")
    return p


def require_model_dir(model_key: str = "v2") -> Path:
    """确认模型目录存在且包含完整的 yaml+ckpt 对。
    
    Args:
        model_key: 模型版本键，如 "v2"、"v3"。
    
    Returns:
        模型目录路径。
    
    Raises:
        ResourceNotFound: 如果目录不存在或缺少必要文件。
    """
    model_dir = bs_roformer_revive_dir()
    require_dir(model_dir, f"模型根目录 {model_key}")
    
    # 查找对应的 yaml 和 ckpt 文件
    yaml_files = list(model_dir.glob(f"*{model_key}*.yaml"))
    ckpt_files = list(model_dir.glob(f"*{model_key}*.ckpt"))
    
    if not yaml_files:
        raise ResourceNotFound(
            f"缺少模型 {model_key} 的配置文件 (*.yaml)：{model_dir}\n"
            f"可用文件：{[f.name for f in model_dir.iterdir() if f.suffix in ('.yaml', '.yml', '.ckpt')]}")
    
    if not ckpt_files:
        raise ResourceNotFound(
            f"缺少模型 {model_key} 的权重文件 (*.ckpt)：{model_dir}\n"
            f"可用文件：{[f.name for f in model_dir.iterdir() if f.suffix in ('.yaml', '.yml', '.ckpt')]}")
    
    # 返回包含匹配 yaml 和 ckpt 的目录
    return model_dir


@dataclass
class ModelInfo:
    """模型信息。"""
    key: str
    yaml_path: Path
    ckpt_path: Path


def list_models() -> list[ModelInfo]:
    """扫描 `models/bs_roformer_revive/` 下所有成对的 yaml+ckpt。
    
    只返回**完整的配对**（同一前缀名同时有 .yaml 和 .ckpt）。
    
    Returns:
        ModelInfo 列表，每个包含 key（如 "v2"）、yaml_path、ckpt_path。
    """
    model_dir = bs_roformer_revive_dir()
    
    if not model_dir.is_dir():
        return []
    
    # 收集所有 yaml 文件
    yaml_files = {}
    for f in model_dir.glob("*.yaml"):
        # 尝试提取模型键（如 "v2"、"v3"）
        stem = f.stem
        for key in ("v2", "v3"):
            if key in stem.lower():
                yaml_files[key] = f
                break
    
    # 检查哪些有对应的 ckpt
    models = []
    for key, yaml_file in yaml_files.items():
        # 查找同名 ckpt（相同前缀）
        stem = yaml_file.stem
        ckpt_file = model_dir / f"{stem}.ckpt"
        if ckpt_file.exists():
            models.append(ModelInfo(key=key, yaml_path=yaml_file, ckpt_path=ckpt_file))
    
    return models


def reset_cache() -> None:
    """清探测缓存（测试用：改环境变量后要重新探测）。"""
    _CACHE.clear()


__all__ = [
    "ENV_ROOT", "ENV_OUTPUT", "ROOT_MARKERS", "ResourceNotFound",
    "repo_root", "diagnose", "models_dir", "ffmpeg_dir", "ffmpeg_bin",
    "ffprobe_bin", "find_ff", "output_dir", "test_audio_dir", "test_clips_dir",
    "self_executable", "bs_roformer_revive_dir",
    "require", "require_dir", "require_model_dir", "list_models",
    "reset_cache",
]

if __name__ == "__main__":
    # Windows GBK 下打印 ✔/✗ 会崩，强制 UTF-8
    if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    print(diagnose())
    print(f"\nPython  {sys.version.split()[0]}")
    print(f"模型    {models_dir()}")
    print(f"ffmpeg  {ffmpeg_bin()}  存在={ffmpeg_bin().exists()}")
    print(f"ffprobe {ffprobe_bin()}  存在={ffprobe_bin().exists()}")
    print(f"产物    {output_dir()}")
    print(f"素材    {test_audio_dir()}  存在={test_audio_dir().exists()}")
    print(f"切片    {test_clips_dir()}  存在={test_clips_dir().exists()}")
    print(f"Python  {self_executable()}")
    print(f"可用模型: {[m.key for m in list_models()]}")