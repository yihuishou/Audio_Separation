# -*- coding: utf-8 -*-
"""主窗口：左侧参数面板，右侧批量队列，底部状态栏。

★ 这里是「参数面板 ↔ 批量队列 ↔ 后台 worker」三者唯一的连接点：
  - `BatchPanel.start_requested` → 收集 ParamPanel 的真实参数 → 起 `BatchWorker`
  - `BatchWorker` 的事件信号 → 回灌 BatchPanel 的日志与进度
  - `BatchPanel.cancel_requested` → `BatchWorker.request_cancel()`
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QStatusBar, QLabel, QHBoxLayout
)

from .batch_panel import BatchPanel
from .param_panel import ParamPanel
from .workers import BatchWorker


class MainWindow(QMainWindow):
    """BS-RoFormer 音声分离工具主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("音声分离 - BS-RoFormer Revive")
        self.resize(1200, 800)
        # 去掉标题栏上的 ? 帮助按钮。
        #
        # 这里必须写成 `& ~int(hint)` 而不是 `& ~hint`：Qt.WindowType 是 IntFlag,
        # 而 IntFlag.__invert__ 只在**已定义成员的并集**内取反 —— PySide6 里
        # ~Qt.WindowContextHelpButtonHint == 0x1feffff，这个掩码覆盖不到
        # WindowCloseButtonHint(0x8000000) / WindowMinimizeButtonHint(0x20000),
        # `&` 之后它们会被当作 0 清掉，表现为标题栏的关闭/最小化/最大化按钮消失。
        # 先 int() 取出真实位掩码再取反，才是全域按位取反。
        _flags = int(self.windowFlags()) & ~int(Qt.WindowType.WindowContextHelpButtonHint)
        self.setWindowFlags(Qt.WindowType(_flags))

        self._worker: BatchWorker | None = None
        self._elapsed_sec = 0
        # 本轮结果：ok / cancelled / failed。决定进度条是否跳 100%、状态栏写什么。
        self._outcome = "ok"

        self._setup_menu()
        self._setup_ui()
        self._setup_status_bar()
        self._setup_timer()

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
        self.param_panel = ParamPanel()
        splitter.addWidget(self.param_panel)

        # 右侧：批量队列（占 3/4）
        self.batch_panel = BatchPanel()
        splitter.addWidget(self.batch_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)
        self.setCentralWidget(central)

        # ── 接线 ────────────────────────────────────────────────────────
        self.batch_panel.start_requested.connect(self._on_start_requested)
        self.batch_panel.cancel_requested.connect(self._on_cancel_requested)
        self.batch_panel.files_added.connect(self._on_files_added)

    def _setup_status_bar(self) -> None:
        """底部状态栏：左侧状态标签，右侧耗时计时器。"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label, 1)

        self.elapsed_label = QLabel()
        self.status_bar.addPermanentWidget(self.elapsed_label)

    def _setup_timer(self) -> None:
        """处理期间的耗时计时器（1 秒一跳）。"""
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)

    # ── 处理流程 ──────────────────────────────────────────────────────────

    def _on_start_requested(self, payload: dict) -> None:
        """开始处理：收集真实参数，起后台 worker。

        ★参数一律从 `ParamPanel.get_params()` 取 —— 面板里不再硬编码
          任何值，否则左边的模型/精度/输出目录选择全是摆设。
        """
        if self._worker is not None and self._worker.isRunning():
            self.batch_panel.show_error("已有任务在运行")
            return

        files = payload.get("files", [])
        if not files:
            self.batch_panel.show_error("没有待处理文件")
            return

        params = self.param_panel.get_params()
        output_dir = params.get("output_dir") or ""

        # ★output_dir 是**根目录**：worker 内部会再拼上每个文件的 stem
        #   （`output_dir_for()` → `<root>/<曲名>/`），与 CLI 行为一致。
        inputs = [{"src": f, "output_dir": output_dir} for f in files]

        self._elapsed_sec = 0
        self.elapsed_label.setText("00:00")
        self.status_label.setText(f"处理中（{len(files)} 个文件）...")
        self._timer.start()
        self._outcome = "ok"

        worker = BatchWorker(
            inputs,
            model_key=params.get("model_key") or "v2",
            device=params.get("device") or "auto",
            dtype=params.get("dtype") or "float32",
            parent=self,
        )
        worker.log_line.connect(self.batch_panel.append_log)
        worker.progress_updated.connect(self._on_progress)
        worker.file_finished.connect(self._on_file_finished)
        worker.fatal_error.connect(self._on_fatal)
        worker.run_cancelled.connect(self._on_cancelled)
        worker.batch_finished.connect(self._on_batch_finished)
        worker.finished.connect(self._on_worker_finished)

        self._worker = worker
        worker.start()

    def _on_cancel_requested(self) -> None:
        """取消处理。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self.status_label.setText("正在取消...")

    def _on_files_added(self, files: list) -> None:
        """文件列表变化时更新状态栏。"""
        if self._worker is None or not self._worker.isRunning():
            self.status_label.setText(f"已添加 {len(files)} 个文件")

    def _on_progress(self, current: int, total: int, message: str, pct: int) -> None:
        """进度更新。"""
        self.batch_panel.update_progress(current, total, message, pct)

    def _on_file_finished(self, name: str, ok: bool, out_dir: str) -> None:
        """单个文件完成。"""
        if ok:
            self.batch_panel.append_log(f"[完成] {name} → {out_dir}")
        else:
            self.batch_panel.append_log(f"[失败] {name}")

    def _on_fatal(self, message: str) -> None:
        """致命错误。"""
        self._outcome = "failed"
        self.batch_panel.show_error(message)
        self.status_label.setText("处理失败")

    def _on_cancelled(self, reason: str) -> None:
        """任务被取消。"""
        self._outcome = "cancelled"
        self.batch_panel.append_log(f"[取消] {reason}")
        self.status_label.setText("已取消")

    def _on_batch_finished(self, summary: dict) -> None:
        """整个队列结束。"""
        total = summary.get("total", 0)
        ok = summary.get("ok", 0)
        failed = summary.get("failed", 0)
        cancelled = summary.get("cancelled", 0)
        if cancelled:
            self._outcome = "cancelled"
        if self._outcome != "failed":
            self.batch_panel.append_log(
                f"汇总：共 {total} 个，成功 {ok}，失败 {failed}，取消 {cancelled}"
            )
        if self._outcome == "ok":
            self.status_label.setText("处理完成" if not failed else f"处理完成（{failed} 个失败）")

    def _on_worker_finished(self) -> None:
        """worker 线程退出（正常结束或异常）。"""
        self._timer.stop()
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        # 面板复位（出错/取消路径已由 show_error 复位，这里再兜一次底）
        if self.batch_panel.processing:
            self.batch_panel.finish_processing(complete=self._outcome == "ok")

    def _on_tick(self) -> None:
        """耗时计时。"""
        self._elapsed_sec += 1
        minutes, seconds = divmod(self._elapsed_sec, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            self.elapsed_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.elapsed_label.setText(f"{minutes:02d}:{seconds:02d}")

    # ── 菜单 ──────────────────────────────────────────────────────────────

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

    def closeEvent(self, event) -> None:
        """关闭时先让 worker 停下来。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(5000)
        super().closeEvent(event)

    @property
    def status(self) -> QLabel:
        return self.status_label
