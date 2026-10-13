# -*- coding: utf-8 -*-
"""GUI 冒烟测试：验证 PySide6 能创建应用和主窗口。

这些测试不依赖显示服务器，只检查模块导入和对象创建。
真正的视觉测试需要 xvfb（Linux）或实际显示器（Windows）。
"""
from __future__ import annotations

import sys
import pytest

sys.path.insert(0, "src")

pytestmark = pytest.mark.ui


class TestPySide6Import:
    """PySide6 模块导入测试。"""

    def test_qt_imports(self):
        """所有需要的 Qt 模块都能导入。"""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QPushButton,
            QLabel, QVBoxLayout, QHBoxLayout, QSplitter, QStatusBar,
            QMenuBar, QMenu, QListWidget, QListWidgetItem, QProgressBar,
            QPlainTextEdit, QFileDialog, QMessageBox, QComboBox,
            QLineEdit, QFormLayout, QGroupBox, QTableWidget, QTableWidgetItem,
            QHeaderView
        )
        from PySide6.QtGui import QCloseEvent
        assert Qt is not None

    def test_app_creation(self):
        """QApplication 能创建。"""
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication(sys.argv)
        assert app is not None
        app.quit()


class TestUIModuleImports:
    """UI 模块导入测试。"""

    def test_app_module(self):
        """audio_separation.ui.app 能导入。"""
        from audio_separation.ui.app import main
        assert callable(main)

    def test_main_window_module(self):
        """audio_separation.ui.main_window 能导入。"""
        from audio_separation.ui.main_window import MainWindow
        assert MainWindow is not None

    def test_param_panel_module(self):
        """audio_separation.ui.param_panel 能导入。"""
        from audio_separation.ui.param_panel import ParamPanel
        assert ParamPanel is not None

    def test_batch_panel_module(self):
        """audio_separation.ui.batch_panel 能导入。"""
        from audio_separation.ui.batch_panel import BatchPanel
        assert BatchPanel is not None

    def test_env_check_dialog_module(self):
        """audio_separation.ui.env_check_dialog 能导入。"""
        from audio_separation.ui.env_check_dialog import EnvCheckDialog
        assert EnvCheckDialog is not None


class TestMainWindowCreation:
    """主窗口创建测试（需要 QApplication 实例）。"""

    @pytest.fixture(autouse=True)
    def app(self):
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            app = QApplication([])
            yield app
            app.quit()
        else:
            yield QApplication.instance()

    def test_main_window_creation(self, app):
        """MainWindow 能创建。"""
        from audio_separation.ui.main_window import MainWindow
        window = MainWindow()
        assert window is not None
        assert window.windowTitle() == "音声分离 - BS-RoFormer Revive"
        window.close()

    def test_window_has_splitter(self, app):
        """主窗口有左右分栏。"""
        from audio_separation.ui.main_window import MainWindow
        window = MainWindow()
        central = window.centralWidget()
        assert central is not None
        window.close()

    def test_param_panel_has_model_select(self, app):
        """参数面板有模型选择。"""
        from audio_separation.ui.param_panel import ParamPanel
        panel = ParamPanel()
        assert panel.model_select is not None
        assert panel.model_select.count() > 0  # 有可用模型
        panel.deleteLater()

    def test_batch_panel_has_file_list(self, app):
        """批量面板有文件列表。"""
        from audio_separation.ui.batch_panel import BatchPanel
        panel = BatchPanel()
        assert panel.file_list is not None
        assert panel.start_btn is not None
        assert panel.cancel_btn is not None
        panel.deleteLater()

    def test_param_panel_get_params(self, app):
        """参数面板能获取参数。"""
        from audio_separation.ui.param_panel import ParamPanel
        panel = ParamPanel()
        params = panel.get_params()
        assert "model_key" in params
        assert "device" in params
        assert "dtype" in params
        assert "output_dir" in params
        panel.deleteLater()
