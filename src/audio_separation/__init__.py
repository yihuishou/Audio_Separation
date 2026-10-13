"""audio_separation - BS-RoFormer 音声分离图形化工具

★ 这里必须在任何 librosa / torch / bs_roformer 的 import **之前**执行启动期规避
（PLAN §2.8）。放在 `__init__.py` 是因为它是唯一保证先于所有子模块执行的位置：
CLI、worker 子进程、tools/ 脚本不管从哪个入口进来，都会先经过这里。
把规避写进每个入口迟早会漏掉一个，而漏掉的那一个表现为**静默死循环**——
没有异常、没有日志、进程只是永远不返回。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"

__all__ = ["__version__", "numba_cache_dir"]


def numba_cache_dir() -> Path | None:
    """把 NUMBA_CACHE_DIR 指到一个可写目录并返回它；无法建立则返回 None。

    为什么需要它：`bs_roformer` 顶层 import 了 `librosa`，而 librosa 的 numba
    装饰器在**模块执行期**就要落缓存。默认位置是 site-packages 的 `__pycache__`；
    该目录不可写时 numba 会在里面反复尝试建临时文件，**不抛异常、不返回** ——
    实测表现为进程静默挂死 50 分钟以上，栈顶固定停在
    `tempfile._mkstemp_inner` ← `numba.core.caching.ensure_cache_path`。
    """
    if existing := os.environ.get("NUMBA_CACHE_DIR"):
        return Path(existing)

    base = os.environ.get("TEMP") or os.environ.get("TMP")
    root = Path(base) if base else Path.home() / ".cache"
    target = root / "audio_separation_numba_cache"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 落回默认行为：由 librosa/numba 自己决定往哪写。
        # 这里刻意不抛 —— 环境自检层（§2.6）会把真实后果讲清楚，
        # 而让"包一导入就崩"会连诊断信息都拿不到。
        return None
    os.environ["NUMBA_CACHE_DIR"] = str(target)
    return target


def _reconfigure_streams() -> None:
    """stdout/stderr 改成 UTF-8 且不因编码失败而崩（PLAN §2.8 规避②）。

    Windows 控制台默认 GBK，而本项目的日志里固定会出现 `·` `⇒` `★` 与
    日文/全角字符；一旦被重定向到文件（CLI 的常态）就会 UnicodeEncodeError。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # 被替换成了没有 reconfigure 的对象（如某些测试里的 StringIO），
            # 或者流已关闭 —— 都不是错误
            pass


numba_cache_dir()
_reconfigure_streams()
