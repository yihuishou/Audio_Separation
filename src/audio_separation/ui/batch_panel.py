# -*- coding: utf-8 -*-
"""批量队列面板（右侧）：文件列表、传输控制、进度显示、日志。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QProgressBar, QPlainTextEdit, QFileDialog,
    QMessageBox
)

from .. import paths
from ..core import batch


class BatchPanel(QWidget):
    """批量队列面板。"""

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
            import os
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

    def _on_start(self) -> None:
        """开始处理。"""
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            QMessageBox.warning(self, "警告", "请先添加文件")
            return

        self._processing = True
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log.append("开始处理...")

        # 获取参数（从主窗口传入或默认）
        params = {"model_key": "v2", "device": "auto", "dtype": "float32"}
        self.start_requested.emit({"files": files, **params})

    def _on_cancel(self) -> None:
        """取消处理。"""
        self.cancel_requested.emit()
        self.log.append("取消处理...")

    def add_files(self, files: list[str]) -> None:
        """添加文件到列表。"""
        for file in files:
            item = QListWidgetItem(file)
            self.file_list.addItem(item)

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """更新进度。"""
        if total > 0:
            pct = int(100 * current / total)
            self.progress.setValue(pct)
            self.file_label.setText(f"{current} / {total} 文件")
        if message:
            self.log.append(message)

    def finish_processing(self) -> None:
        """处理完成。"""
        self._processing = False
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100)
        self.log.append("处理完成")
        self.processing_complete.emit()

    @property
    def processing(self) -> bool:
        return self._processing
