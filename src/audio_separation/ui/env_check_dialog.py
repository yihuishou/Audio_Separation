# -*- coding: utf-8 -*-
"""环境自检对话框。"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView
)

from ..core.env_check import check_libs, probe_device, build_report_rows


class EnvCheckWorker(QThread):
    """后台环境检查 worker。"""
    finished = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self._rows = []

    def run(self) -> None:
        try:
            self._rows = build_report_rows()
        except Exception:
            pass
        self.finished.emit(self._rows)


class EnvCheckDialog(QDialog):
    """环境自检对话框。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("环境自检")
        self.setModal(True)
        self.resize(800, 600)
        self._setup_ui()
        self._run_check()

    def _setup_ui(self) -> None:
        """构建 UI。"""
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["检查项", "级别", "详情", "修复建议"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _run_check(self) -> None:
        """运行环境检查。"""
        self.worker = EnvCheckWorker()
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, rows: list) -> None:
        """检查结果填充表格。"""
        self.table.setRowCount(len(rows))
        for row, r in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(r["item"]))
            level = QTableWidgetItem(r["level"])
            level.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, level)
            self.table.setItem(row, 2, QTableWidgetItem(r["detail"]))
            remedy = QTableWidgetItem(r.get("remedy", ""))
            if r["level"] in ("FAIL", "WARN"):
                remedy.setForeground(Qt.red)
            self.table.setItem(row, 3, remedy)
