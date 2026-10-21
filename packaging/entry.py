# -*- coding: utf-8 -*-
"""PyInstaller 的 GUI 入口 stub。

★ 为什么不能直接拿 ``src/audio_separation/ui/app.py`` 当入口：
PyInstaller 以 script 方式运行入口文件，此时 ``__package__`` 为空，
``app.py`` 里的 ``from ..core.protocol import WORKER_FLAG`` 这种相对 import
会直接抛 ``ImportError: attempted relative import with no known parent package``。

所以这里用绝对 import 包一层。入口文件名不能叫 ``__main__.py`` ——
那会和包内的 ``src/audio_separation/__main__.py`` 撞车。
"""
from __future__ import annotations

import sys

from audio_separation.ui.app import main

if __name__ == "__main__":
    sys.exit(main())
