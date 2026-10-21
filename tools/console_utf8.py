# -*- coding: utf-8 -*-
"""把 stdout/stderr 切到 UTF-8，避免在非 UTF-8 控制台上打印中文时崩掉。

★ 为什么要专门做这件事：Windows 上 Python 的标准输出编码跟着
   **控制台代码页**走。CI 的 Windows runner 是英文环境（cp1252），
   打印中文直接抛：

       UnicodeEncodeError: 'charmap' codec can't encode characters ...

   最冤的是脚本逻辑其实已经跑完了 —— 版本号已经写进文件，
   却因为最后一句 print 失败而返回非 0 退出码，把整条流水线带崩。

   改成 UTF-8 输出比改成英文输出好：日志里保留中文，排查时一眼能看懂。
   ``errors="replace"`` 是最后一道保险 —— 万一目标流连 UTF-8 都接不住，
   宁可显示成问号，也不要抛异常。
"""

from __future__ import annotations

import sys


def force_utf8() -> None:
    """把 stdout/stderr 切成 UTF-8，输出不了就替换而不是抛异常。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 某些流（被换成文件、被 pytest 捕获等）不支持 reconfigure
            pass
