# -*- coding: utf-8 -*-
"""批量队列面板（右侧）：文件列表、传输控制、进度显示、日志。"""
from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QProgressBar, QPlainTextEdit, QFileDialog,
    QMessageBox
)


class BatchPanel(QWidget):
    """批量队列面板。

    ★本面板只管「界面状态」，不碰推理：`_on_start` 只发 `start_requested`，
      真正的调度由 `MainWindow` 接的 `BatchWorker`（ui/workers.py）负责。
      面板自己不 import torch，GUI 进程始终不加载重依赖（§2.3）。
    """

    # 信号：文件添加、开始处理、取消、完成
    files_added = Signal(list)
    start_requested = Signal(dict)
    cancel_requested = Signal()
    processing_complete = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()
        self._processing = False

    def _setup_ui(self) -> None:
        """构建批量队列 UI。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 文件列表组
        files_group = QVBoxLayout()
        files_group.addWidget(QLabel("输入文件："))

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        files_group.addWidget(self.file_list, 1)

        # 添加/移除按钮
        file_btns = QHBoxLayout()

        add_btn = QPushButton("添加文件(&A)...")
        add_btn.clicked.connect(self._add_files)
        file_btns.addWidget(add_btn)

        add_dir_btn = QPushButton("添加文件夹(&D)...")
        add_dir_btn.clicked.connect(self._add_directory)
        file_btns.addWidget(add_dir_btn)

        remove_btn = QPushButton("移除选中(&R)")
        remove_btn.clicked.connect(self._remove_selected)
        file_btns.addWidget(remove_btn)

        files_group.addLayout(file_btns)
        layout.addLayout(files_group, 1)

        # 传输控制组
        transport = QHBoxLayout()

        self.start_btn = QPushButton("开始处理(&S)")
        self.start_btn.clicked.connect(self._on_start)
        transport.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("取消(&C)")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        transport.addWidget(self.cancel_btn)

        layout.addLayout(transport)

        # 进度组
        progress_group = QVBoxLayout()

        self.file_label = QLabel("0 / 0 文件")
        progress_group.addWidget(self.file_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        progress_group.addWidget(self.progress)

        layout.addLayout(progress_group)

        # 日志组
        log_group = QVBoxLayout()
        log_group.addWidget(QLabel("日志："))

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        # ★只保留最近 500 行（PLAN §4 M4-6）：全曲 15 分钟会刷出上万行，
        #   无上限的话 QPlainTextEdit 内存暴涨 → GUI 卡死。
        self.log.setMaximumBlockCount(500)
        log_group.addWidget(self.log, 1)

        layout.addLayout(log_group, 1)

    def _add_files(self) -> None:
        """添加音频文件。"""
        audio_filters = (
            "音频文件 (*.mp3 *.flac *.wav *.m4a *.ogg *.wma);;"
            "所有文件 (*.*)"
        )
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择音频文件", "", audio_filters
        )
        if files:
            self.add_files(files)

    def _add_directory(self) -> None:
        """添加文件夹中的所有音频文件。"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择文件夹"
        )
        if directory:
            audio_ext = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".wma"}
            files = []
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    if any(filename.lower().endswith(ext) for ext in audio_ext):
                        files.append(os.path.join(root, filename))
            if files:
                self.add_files(files)

    def _remove_selected(self) -> None:
        """移除选中的文件。"""
        items = self.file_list.selectedItems()
        for item in items:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
        self._refresh_file_label()

    def _on_start(self) -> None:
        """开始处理。

        ★只发信号，不自己干活。参数（模型/精度/输出目录）由 `MainWindow`
          从 `ParamPanel` 取真实值后补齐 —— 面板里不硬编码任何参数，
          否则左边的参数面板就是个纯装饰。
        """
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            QMessageBox.warning(self, "警告", "请先添加文件")
            return

        self._processing = True
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.append_log("开始处理...")

        self.start_requested.emit({"files": files})

    def _on_cancel(self) -> None:
        """取消处理。"""
        self.cancel_requested.emit()
        self.append_log("取消处理...")

    def add_files(self, files: list[str]) -> None:
        """添加文件到列表。"""
        for file in files:
            item = QListWidgetItem(file)
            self.file_list.addItem(item)
        self._refresh_file_label()
        self.files_added.emit(list(files))

    def _refresh_file_label(self) -> None:
        """按列表当前条数刷新「N / M 文件」标签。"""
        total = self.file_list.count()
        self.file_label.setText(f"0 / {total} 文件")

    def append_log(self, text: str) -> None:
        """往日志框追加一行。

        ★`QPlainTextEdit` 没有 `append()` 方法（那是 `QTextEdit` 的 API），
          正确写法是 `appendPlainText()`。旧的 `self.log.append(...)` 会抛
          `AttributeError: 'QPlainTextEdit' object has no attribute 'append'`。
        """
        self.log.appendPlainText(str(text))

    def update_progress(
        self, current: int, total: int, message: str = "", pct: int | None = None
    ) -> None:
        """更新进度。

        Args:
            current: 已完成的文件数。
            total: 文件总数。
            message: 附加到日志的一行说明（可空）。
            pct: 整体百分比。给了就用它（分块级进度比文件级更细），
                 不给则按文件数算。
        """
        if total > 0:
            self.file_label.setText(f"{current} / {total} 文件")
            if pct is None:
                pct = int(100 * current / total)
            self.progress.setValue(max(0, min(100, pct)))
        if message:
            self.append_log(message)

    def show_error(self, message: str) -> None:
        """在日志里报一条错误，并把界面切回「可开始」状态。"""
        self.append_log(f"[错误] {message}")
        self.finish_processing(complete=False)

    def finish_processing(self, complete: bool = True) -> None:
        """处理结束。

        Args:
            complete: 是否正常完成。取消/出错时进度条不该跳到 100% ——
                那会让「没做完」和「做完了」在界面上长得一模一样。
        """
        self._processing = False
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if complete:
            self.progress.setValue(100)
            self.append_log("处理完成")
        self.processing_complete.emit()

    @property
    def processing(self) -> bool:
        return self._processing
