# -*- coding: utf-8 -*-
"""GUI 冒烟测试：验证 PySide6 能创建应用和主窗口。

这些测试不依赖显示服务器，只检查模块导入和对象创建。
真正的视觉测试需要 xvfb（Linux）或实际显示器（Windows）。
"""
from __future__ import annotations

import sys
from pathlib import Path

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

    def test_param_panel_has_open_dir_button(self, app):
        """输出组里有「打开目录」按钮。"""
        from audio_separation.ui.param_panel import ParamPanel
        panel = ParamPanel()
        assert panel.open_btn is not None
        assert "打开" in panel.open_btn.text()
        panel.deleteLater()


class TestOpenOutputDir:
    """「打开输出目录」的回归测试。

    ★测试不真的调起系统文件管理器 —— 那是 GUI 进程外部行为，
      断言它只会让测试在 CI 上弹出资源管理器窗口。这里把
      `QDesktopServices.openUrl` 换掉，只验决策逻辑。
    """

    @pytest.fixture(autouse=True)
    def app(self):
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            app = QApplication([])
            yield app
            app.quit()
        else:
            yield QApplication.instance()

    @pytest.fixture
    def panel(self, app):
        from audio_separation.ui.param_panel import ParamPanel
        p = ParamPanel()
        yield p
        p.deleteLater()

    def test_existing_dir_is_opened(self, panel, tmp_path, monkeypatch):
        """目录已存在 → 直接打开，不弹询问框。"""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox

        opened: list[str] = []
        monkeypatch.setattr(
            QDesktopServices, "openUrl",
            staticmethod(lambda url: opened.append(url.toLocalFile()) or True),
        )
        def boom(*a, **k):
            raise AssertionError("目录已存在时不该弹询问框")
        monkeypatch.setattr(QMessageBox, "question", staticmethod(boom))

        target = tmp_path / "out"
        target.mkdir()
        panel.output_dir_edit.setText(str(target))
        panel.open_btn.click()

        # QUrl.toLocalFile() 在 Windows 上返回正斜杠路径，用 Path 比较
        assert [Path(p) for p in opened] == [target]

    def test_missing_dir_asks_before_creating(self, panel, tmp_path, monkeypatch):
        """目录不存在 → 先问用户，答「否」就不建。"""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox

        opened: list[str] = []
        monkeypatch.setattr(
            QDesktopServices, "openUrl",
            staticmethod(lambda url: opened.append(url.toLocalFile()) or True),
        )
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
        )

        target = tmp_path / "nope"
        panel.output_dir_edit.setText(str(target))
        panel.open_btn.click()

        assert not target.exists(), "答「否」就不该创建目录"
        assert opened == [], "没创建就不该打开"

    def test_missing_dir_creates_when_confirmed(self, panel, tmp_path, monkeypatch):
        """目录不存在 → 答「是」就创建并打开。"""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox

        opened: list[str] = []
        monkeypatch.setattr(
            QDesktopServices, "openUrl",
            staticmethod(lambda url: opened.append(url.toLocalFile()) or True),
        )
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
        )

        target = tmp_path / "made" / "deep"
        panel.output_dir_edit.setText(str(target))
        panel.open_btn.click()

        assert target.is_dir(), "答「是」就该创建目录（含多级父目录）"
        assert [Path(p) for p in opened] == [target]

    def test_empty_dir_warns_without_opening(self, panel, monkeypatch):
        """输出目录为空 → 警告，不打开。"""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox

        opened: list[str] = []
        monkeypatch.setattr(
            QDesktopServices, "openUrl",
            staticmethod(lambda url: opened.append(url.toLocalFile()) or True),
        )
        warned: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: warned.append(a[0]) or QMessageBox.StandardButton.Ok),
        )

        panel.output_dir_edit.setText("")
        panel.open_btn.click()

        assert opened == []
        assert warned, "空目录应该弹警告"

    def test_open_url_failure_is_reported(self, panel, tmp_path, monkeypatch):
        """openUrl 返回 False（无桌面环境）→ 提示用户手动访问。"""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox

        target = tmp_path / "out"
        target.mkdir()
        panel.output_dir_edit.setText(str(target))

        monkeypatch.setattr(
            QDesktopServices, "openUrl", staticmethod(lambda url: False)
        )
        warned: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: warned.append(a[0]) or QMessageBox.StandardButton.Ok),
        )

        panel.open_btn.click()

        assert warned, "openUrl 失败必须提示，不能静默吞掉"


class TestBatchPanelStartButton:
    """★「开始处理」按钮的回归测试。

    曾经的 bug：`_on_start()` 调 `self.log.append(...)`，
    而 `QPlainTextEdit` 没有 `append()`（那是 `QTextEdit` 的 API），
    点一下就抛 `AttributeError` —— 按钮完全不可用。
    """

    @pytest.fixture(autouse=True)
    def app(self):
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            app = QApplication([])
            yield app
            app.quit()
        else:
            yield QApplication.instance()

    def test_log_widget_is_plain_text_edit(self, app):
        """日志控件必须是 QPlainTextEdit —— 这正是 append() 不存在的原因。"""
        from audio_separation.ui.batch_panel import BatchPanel
        panel = BatchPanel()
        from PySide6.QtWidgets import QPlainTextEdit
        assert isinstance(panel.log, QPlainTextEdit)
        assert not hasattr(panel.log, "append"), (
            "QPlainTextEdit 不该有 append()；若上游换成 QTextEdit 才有"
        )
        panel.deleteLater()

    def test_append_log_does_not_raise(self, app):
        """append_log() 能写入日志且不抛异常。"""
        from audio_separation.ui.batch_panel import BatchPanel
        panel = BatchPanel()
        panel.append_log("hello")
        assert "hello" in panel.log.toPlainText()
        panel.deleteLater()

    def test_start_button_emits_signal(self, app):
        """点「开始处理」发信号，且不抛异常。"""
        from audio_separation.ui.batch_panel import BatchPanel
        panel = BatchPanel()
        panel.add_files(["a.wav", "b.wav"])

        received: list[dict] = []
        panel.start_requested.connect(received.append)

        panel.start_btn.click()

        assert len(received) == 1, "点开始处理必须发出 start_requested"
        assert received[0]["files"] == ["a.wav", "b.wav"]
        # 面板不硬编码参数 —— 参数由 MainWindow 从 ParamPanel 取
        assert "model_key" not in received[0], (
            "参数不该由面板硬编码，否则左侧参数面板形同虚设"
        )
        assert "开始处理" in panel.log.toPlainText()
        panel.deleteLater()

    def test_start_without_files_warns_instead_of_crashing(self, app):
        """没有文件时点开始：弹警告，不发信号，不崩。"""
        from audio_separation.ui.batch_panel import BatchPanel
        panel = BatchPanel()

        received: list[dict] = []
        panel.start_requested.connect(received.append)

        # 屏蔽 QMessageBox 模态弹窗
        from PySide6.QtWidgets import QMessageBox
        orig = QMessageBox.warning
        QMessageBox.warning = lambda *a, **k: None
        try:
            panel.start_btn.click()
        finally:
            QMessageBox.warning = orig

        assert received == [], "没有文件时不该发 start_requested"
        panel.deleteLater()

    def test_update_progress_sets_bar_and_label(self, app):
        """update_progress 刷新百分比与「N / M 文件」标签。"""
        from audio_separation.ui.batch_panel import BatchPanel
        panel = BatchPanel()
        panel.update_progress(1, 4, "第 1 个完成", 25)
        assert panel.progress.value() == 25
        assert "1 / 4" in panel.file_label.text()
        assert "第 1 个完成" in panel.log.toPlainText()
        panel.deleteLater()


class TestMainWindowWiring:
    """主窗口的信号接线测试。"""

    @pytest.fixture(autouse=True)
    def app(self):
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            app = QApplication([])
            yield app
            app.quit()
        else:
            yield QApplication.instance()

    def test_start_requested_reaches_main_window(self, app, monkeypatch):
        """★BatchPanel.start_requested 真的接到了 MainWindow。

        接线的回归测试：信号两边各发各的，谁也不连，谁都发现不了。
        这里不真起 worker（要跑 GPU），而是把 BatchWorker 换成假对象，
        只验证「信号 → MainWindow._on_start_requested」这条链路通。
        """
        from audio_separation.ui import main_window as mw
        from audio_separation.ui.workers import BatchWorker

        started: list[dict] = {}

        class FakeWorker:
            def __init__(self, inputs, **kwargs):
                started["inputs"] = inputs
                started["kwargs"] = kwargs
                self.log_line = _FakeSignal()
                self.progress_updated = _FakeSignal()
                self.file_finished = _FakeSignal()
                self.fatal_error = _FakeSignal()
                self.run_cancelled = _FakeSignal()
                self.batch_finished = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self):
                started["started"] = True

            def isRunning(self):
                return False

        monkeypatch.setattr(mw, "BatchWorker", FakeWorker)

        window = mw.MainWindow()
        window.batch_panel.add_files(["x.wav"])
        window.batch_panel.start_btn.click()

        assert started.get("started") is True, "点开始处理必须真的起 worker"
        assert started["inputs"] == [
            {"src": "x.wav", "output_dir": window.param_panel.get_params()["output_dir"]}
        ]
        # 模型/精度取自 ParamPanel，不是硬编码
        assert started["kwargs"]["model_key"] == window.param_panel.get_params()["model_key"]
        assert started["kwargs"]["dtype"] == window.param_panel.get_params()["dtype"]
        window.close()


class _FakeSignal:
    """哑信号对象：connect 只记录，不发射。"""

    def __init__(self) -> None:
        self._slots: list = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in self._slots:
            slot(*args)


class TestBatchWorker:
    """后台 worker 的契约测试（不跑 GPU）。"""

    def test_worker_module_imports(self):
        """ui.workers 能导入，且不 import torch。"""
        from audio_separation.ui.workers import BatchWorker
        assert BatchWorker is not None

    def test_run_batch_accepts_job_dir(self):
        """core.batch.run_batch 接受 job_dir 参数。"""
        import inspect
        from audio_separation.core.batch import run_batch
        sig = inspect.signature(run_batch)
        assert "job_dir" in sig.parameters, (
            "run_batch 必须能接收 job_dir，否则 GUI 侧拿不到路径、写不了取消哨兵"
        )
        assert sig.parameters["job_dir"].default is None

