# -*- coding: utf-8 -*-
"""PySide6 应用入口。

所有 import 都放在 main() 里 —— 这是 PySide6 的经典陷阱：如果模块顶层 import
PySide6 而 Python 解释器同时被其他东西用着（比如打包器），会出现各种诡异的
链接器错误。main() 里 import 能确保只在真正要跑 GUI 时才加载。
"""
from __future__ import annotations

import sys


def main() -> int:
    """应用入口。返回退出码（0 = 成功）。"""
    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow
    from .. import __version__

    app = QApplication(sys.argv)
    app.setApplicationName("Audio Separation")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
