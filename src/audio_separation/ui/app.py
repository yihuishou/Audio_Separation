# -*- coding: utf-8 -*-
"""PySide6 应用入口。

所有 import 都放在 main() 里 —— 这是 PySide6 的经典陷阱：如果模块顶层 import
PySide6 而 Python 解释器同时被其他东西用着（比如打包器），会出现各种诡异的
链接器错误。main() 里 import 能确保只在真正要跑 GUI 时才加载。

## ★ 四种启动模式（PLAN §2.3 / M7 打包）

| 命令 | 模式 | 行为 |
|------|------|------|
| `audio_separation` | GUI | 弹窗口 |
| `audio_separation --version` | 版本 | 打印版本号后退出（退出码 0） |
| `audio_separation --worker --job job.json` | worker | 不弹窗口，读 job.json 干活，stdout 吐 JSON Lines |
| `audio_separation --probe` | 探测 | 不弹窗口，探测 GPU，结果写入 `PROBE_RESULT_FILE` 指定的文件 |

★ 后三个开关是 `tests/test_packaging.py` 与 `core/env_check.py` 依赖的公开契约，
别当成实现细节删掉。

**为什么 worker 模式必须和 GUI 共用同一个可执行文件**：冻结态下没有
`python worker.py` 这种跑法 —— worker.py 是被打包进 `_MEIPASS` 的**源码文本**，
不是可执行文件，exe 会把它的路径当成不认识的参数。所以 exe 只能自己认
`--worker`、重新启动自己。`core/batch.py` 的 `run_worker()` 正是这么干的。

**探测模式同理**：冻结态没有 `python -c "<脚本>"` 这种跑法，exe 会把 `-c`
当成不认识的参数而走进 GUI 分支再也不返回 —— 表现为「探测超时（>90s）」。

★ `--version` / `--worker` / `--probe` 三个分支都抢在 `QApplication` 之前返回 ——
worker 与探测模式绝不能创建 QApplication（GUI 进程禁止 import torch，见 PLAN §2.3）。
"""
from __future__ import annotations

import sys


def _run_worker(rest: list[str]) -> int:
    """worker 模式：把参数转交给 `audio_separation.worker.main()`。

    `rest` 已经剥掉了 `--worker` 开关，剩下的（`--job <path>` 等）原样透传，
    由 worker 自己的 argparse 负责。

    延迟 import：worker 模块会拉进 `core.model` / `core.pipeline`（torch 在
    那里是延迟 import，不会在 import 阶段触发），别让只想 `--version` 的
    调用付这个代价。
    """
    from ..worker import main as worker_main

    return worker_main(rest)


def _run_probe() -> int:
    """探测模式：设备探测必须在独立进程里跑（torch 导入有挂死风险）。

    冻结态下这个独立进程就是 exe 自己 —— 探测脚本以源码形式被打进 bundle，
    没法当可执行文件跑，所以只能让 exe 认 `--probe` 再启动自己。
    """
    from ..core.device_probe import main as probe_main

    return probe_main()


def main(argv: list[str] | None = None) -> int:
    """应用入口。返回退出码（0 = 成功）。"""
    args = list(sys.argv[1:] if argv is None else argv)

    from ..core.protocol import PROBE_FLAG, WORKER_FLAG

    # ── worker 模式：必须抢在创建 QApplication 之前分派 ───────────────
    # 冻结态的 exe 同时是 GUI 和 worker（调度器用 `--worker` 重新拉起自己）。
    if WORKER_FLAG in args:
        return _run_worker([a for a in args if a != WORKER_FLAG])

    # ── 探测模式：同样抢在 QApplication 之前 ─────────────────────────
    if PROBE_FLAG in args:
        return _run_probe()

    from .. import __version__

    # ── --version：不弹窗口，打印版本号后退出 ────────────────────────
    if "--version" in args:
        print(__version__)
        return 0

    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Audio Separation")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
