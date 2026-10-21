# -*- coding: utf-8 -*-
"""隐藏控制台窗口（GUI 模式专用）。

## 为什么需要它

打包产物在 Windows 上双击会弹一个黑色控制台窗口，占地方、也吓人。
用户要的是「启动后不显示额外的控制台」。

## ★为什么不能光靠 PyInstaller 的 `console=False`

`console=False` 的 exe 是 **GUI 子系统**程序。Windows 上 GUI 子系统程序
默认**不分配控制台**，`sys.stdout` 拿不到控制台句柄。

但本项目的 worker 模式靠 **stdout 传 JSON Lines 协议**
（`core/protocol.py`），调度器在 `core/batch.py` 用 `stdout=PIPE` 逐行读。
如果 stdout 没了，协议直接断掉 —— GUI 永远等不到结果。

实测（PyInstaller 6.22.3，Windows，onedir）：

    console=False 时，父进程用 stdout=PIPE 拉起的子进程，
    其 stdout 仍然是可用的管道 —— JSON Lines 收发完全正常。

也就是说 **`console=False` 是安全的**。本模块做的是双保险：
在 GUI 分支里额外调 `FreeConsole()`，把「万一还有残留控制台」的情况也清掉。

## ★Linux / macOS 上这个模块是空操作

Linux 与 macOS 都没有「控制台窗口」这回事 —— 从文件管理器双击
不会像 Windows 那样弹黑框，所以这里直接 `return False`。

（Linux 上 PyInstaller 的 `console=False` 干脆被忽略：源码
`building/api.py` 的 `_bootloader_file()` 只在 `is_win or is_darwin`
时才切到 `runw` 版 bootloader，Linux 用的始终是 `run` 版。
macOS 上 `console=False` 生效，但 macOS 版 bootloader 没有把 stdout
重定向到 /dev/null，worker 协议照样通。详见
`packaging/audio_separation.spec` 里 `console=False` 的注释。）

## ★worker / --version / --probe 三个模式绝不能调这里

- worker 的 stdout 是协议通道，FreeConsole 会让它写不出去
- `--version` 的输出要能被命令行读到
- `--probe` 的结果靠 `PROBE_RESULT_FILE` 传，但它的诊断信息走 stderr

所以本模块只由 `ui/app.py` 的 **GUI 分支** 调用。
"""
from __future__ import annotations

import sys


def hide_console() -> bool:
    """隐藏控制台窗口。返回是否真的做了动作。

    - Windows：用 `FreeConsole()` 摘掉本进程的控制台关联。
    - Linux / macOS：没有「控制台窗口」这回事，返回 False。

    ★失败不抛异常 —— 隐藏窗口是锦上添花，绝不能让程序起不来。
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # FreeConsole 失败只说明「本来就没有控制台」（GUI 子系统程序
        # 或从资源管理器启动），不是错误。
        kernel32.FreeConsole()
        return True
    except (ImportError, OSError, AttributeError):
        return False
