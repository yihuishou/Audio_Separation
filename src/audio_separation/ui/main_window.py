# -*- coding: utf-8 -*-
"""主窗口：左侧参数面板，右侧批量队列，底部状态栏。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QStatusBar, QMenuBar,
    QMenu, QLabel, QVBoxLayout, QHBoxLayout
)


class MainWindow(QMainWindow):
    """BS-RoFormer 音声分离工具主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("音声分离 - BS-RoFormer Revive")
        self.resize(1200, 800)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._setup_menu()
        self._setup_ui()
        self._setup_status_bar()

    def _setup_menu(self) -> None:
        """菜单栏：文件、工具、帮助。"""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("文件(&F)")
        file_menu.addAction("退出(&Q)", self.close)

        tools_menu = menu_bar.addMenu("工具(&T)")
        tools_menu.addAction("环境自检(&E)", self._show_env_check)

        help_menu = menu_bar.addMenu("帮助(&H)")
        help_menu.addAction("关于(&A)", self._show_about)

    def _setup_ui(self) -> None:
        """左右分栏：参数面板 | 批量队列。"""
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧：参数面板（占 1/4）
        from .param_panel import ParamPanel
        self.param_panel = ParamPanel()
        splitter.addWidget(self.param_panel)

        # 右侧：批量队列（占 3/4）
        from .batch_panel import BatchPanel
        self.batch_panel = BatchPanel()
        splitter.addWidget(self.batch_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _setup_status_bar(self) -> None:
        """底部状态栏：左侧状态标签，右侧耗时计时器。"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label, 1)

        self.elapsed_label = QLabel()
        self.status_bar.addPermanentWidget(self.elapsed_label)

    def _show_env_check(self) -> None:
        """显示环境自检对话框。"""
        from .env_check_dialog import EnvCheckDialog
        dialog = EnvCheckDialog(self)
        dialog.exec()

    def _show_about(self) -> None:
        """关于对话框。"""
        from PySide6.QtWidgets import QMessageBox
        from .. import __version__

        QMessageBox.about(
            self, "关于 音声分离",
            f"<h3>音声分离工具 v{__version__}</h3>"
            "<p>基于 BS-RoFormer Revive 模型的 GPU 人声/伴奏分离工具。</p>"
            "<p>模型：bs_roformer_revive v2/v3</p>"
        )

    @property
    def status(self) -> QLabel:
        return self.status_label
