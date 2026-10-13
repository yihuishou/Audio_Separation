# -*- coding: utf-8 -*-
"""分块看门狗（PLAN §4 M3 任务 4）。

## ★ 为什么必须有这个东西

本机 GPU 是 AMD RX 7900 XTX，torch 走 ROCm。ROCm 的算子一旦挂死，
调用栈停在 C++ 里**不返回** —— `KeyboardInterrupt`、`sys.exit()`、
`terminate()` 信号全都无效。参考脚本（`tools/_baseline_separate.py`）
对此的应对是让看门狗线程直接 `os._exit(3)`：进程没了，显存自然释放，
父进程还能从退出码 3 读出「这一轮是超时，不是正常结束」。

**代价是这个模块只允许在推理子进程里用。** 在 GUI 主进程里开，
超时会把整个界面一起杀掉。所以 `core/separator.py` 里的
`separate_vocals()` 接收 `block_timeout_sec` 参数，由调用方
（`cli.py` 进程内、`worker.py` 子进程内）自己决定是否传。

## 两级用法

| 场景 | 动作 |
|---|---|
| GUI 经 `worker.py` 跑（正常路径） | 看门狗**不**自己杀进程，只把诊断写 stderr 并以退出码 3 退出 |
| 直接 `python -m audio_separation.core.cli`（无父进程兜底） | 默认动作 = 打印诊断 + `os._exit(3)`，否则终端里只能 Ctrl+C |

两者用 `on_timeout=` 区分，见 `cli.py` 与 `worker.py`。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

#: 看门狗判定「超时」用的退出码。父进程靠它区分超时与正常结束。
TIMEOUT_EXIT_CODE = 3


def vram_text() -> str:
    """显存占用文本。取不到就返回占位符，绝不让看门狗自己因取显存失败而崩。"""
    try:
        import torch

        return (
            f"显存 已分配 {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB"
            f" / 峰值 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"
        )
    except Exception:
        return "显存信息不可用"


def timeout_report(stage: str, idle: float, limit: float) -> str:
    """超时诊断文本。UI 与 CLI 共用同一份措辞，用户两边看到的解释一致。"""
    return (
        f"阶段「{stage}」已 {idle:.0f}s 无进度（阈值 {limit:.0f}s）\n"
        f"  {vram_text()}\n"
        f"  ROCm/CUDA 算子挂死时无法从 Python 侧中断，已强制结束进程"
        f"（退出码 {TIMEOUT_EXIT_CODE}）。\n"
        f"  建议：关闭其他占用显存的程序、检查 GPU 驱动与 ROCm 支持列表、"
        f"或调大 --block-timeout。\n"
        f"  ★本工具不做 CPU 回退（PLAN §2.14），所以这里只能给排查方向，不能给绕过办法。"
    )


class BlockWatchdog:
    """检测「距上次进度超过 N 秒无进展」。

    Args:
        limit_sec: 阈值秒数，必须为正；传 0 表示不启用（由调用方决定）。
        on_timeout: 超时动作，签名 `(stage, idle_sec) -> None`。
            默认是「写 stderr + `os._exit(3)`」。
    """

    def __init__(
        self,
        limit_sec: float,
        *,
        on_timeout: Callable[[str, float], None] | None = None,
    ) -> None:
        if limit_sec <= 0:
            raise ValueError("limit_sec 必须为正")
        self.limit_sec = float(limit_sec)
        self._on_timeout = on_timeout or self._default_timeout
        self._last = time.monotonic()
        self._stage = "启动"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="block-watchdog", daemon=True)

    @property
    def stage(self) -> str:
        """当前阶段名（供诊断面板展示）。"""
        return self._stage

    def beat(self, stage: str) -> None:
        """报告「进入了新阶段且有进展」。"""
        self._stage = stage
        self._last = time.monotonic()

    def start(self) -> "BlockWatchdog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def __enter__(self) -> "BlockWatchdog":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            idle = time.monotonic() - self._last
            if idle >= self.limit_sec:
                self._on_timeout(self._stage, idle)

    def _default_timeout(self, stage: str, idle: float) -> None:
        import sys

        print(timeout_report(stage, idle, self.limit_sec), file=sys.stderr, flush=True)
        sys.stderr.flush()
        os._exit(TIMEOUT_EXIT_CODE)


__all__ = ["BlockWatchdog", "TIMEOUT_EXIT_CODE", "timeout_report", "vram_text"]
