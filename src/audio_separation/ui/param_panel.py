# -*- coding: utf-8 -*-
"""参数面板（左侧）：模型选择、设备、精度、输出目录。

大部分参数是只读的（模型常量），只有设备、精度、输出目录可改。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QComboBox, QLineEdit, QPushButton
)

from .. import paths
from ..core.config import describe_config, load_config_by_key


class ParamPanel(QWidget):
    """参数面板。"""

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()
        self._load_models()

    def _setup_ui(self) -> None:
        """构建参数面板 UI。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 模型选择组
        model_group = QGroupBox("模型")
        model_layout = QFormLayout(model_group)

        self.model_select = QComboBox()
        self.model_select.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addRow("模型版本", self.model_select)

        from ..core import presets as preset_mod
        preset_row = QHBoxLayout()
        self.preset_select = QComboBox()
        self.preset_select.currentIndexChanged.connect(self._on_preset_changed)
        self._preset_mod = preset_mod
        preset_row.addWidget(self.preset_select, 1)
        self._load_presets()
        model_layout.addRow("预设", preset_row)

        # 模型信息（只读）
        self.model_info = QLabel("")
        self.model_info.setWordWrap(True)
        self.model_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        model_layout.addRow(self.model_info)

        layout.addWidget(model_group)

        # 设备与精度组
        device_group = QGroupBox("设备与精度")
        device_layout = QFormLayout(device_group)

        self.device_label = QLabel("探测中...")
        device_layout.addRow("设备", self.device_label)

        dtype_options = ["float32", "float16"]
        self.dtype_select = QComboBox()
        self.dtype_select.addItems(dtype_options)
        device_layout.addRow("精度", self.dtype_select)

        layout.addWidget(device_group)

        # 输出目录组
        output_group = QGroupBox("输出")
        output_layout = QVBoxLayout(output_group)

        output_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        output_row.addWidget(self.output_dir_edit, 1)

        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_output)
        output_row.addWidget(browse_btn)

        output_layout.addLayout(output_row)
        layout.addWidget(output_group)

        layout.addStretch()

        # 默认输出目录
        self.output_dir_edit.setText(str(paths.output_dir()))

    def _load_models(self) -> None:
        """加载可用模型列表。"""
        for model in paths.list_models():
            self.model_select.addItem(model.key, model.key)

        if self.model_select.count() > 0:
            self._on_model_changed(0)

    def _load_presets(self) -> None:
        self.preset_select.clear()
        presets = self._preset_mod.load_presets()
        for name in presets:
            self.preset_select.addItem(name, name)

    def _on_preset_changed(self, index: int) -> None:
        if index < 0:
            return
        name = self.preset_select.currentData()
        presets = self._preset_mod.load_presets()
        if name in presets:
            params = self._preset_mod.apply_preset(presets[name])
            for i in range(self.model_select.count()):
                if self.model_select.itemData(i) == params["model_key"]:
                    self.model_select.setCurrentIndex(i)
                    break
            for i in range(self.dtype_select.count()):
                if self.dtype_select.itemText(i) == params["dtype"]:
                    self.dtype_select.setCurrentIndex(i)
                    break
            if params["output_dir"]:
                self.output_dir_edit.setText(params["output_dir"])

    def _on_model_changed(self, index: int) -> None:
        """模型选择变化时更新信息。"""
        if index < 0:
            return

        key = self.model_select.currentData()
        try:
            cfg = load_config_by_key(key)
            info = describe_config(cfg)

            lines = [f"{label}: {value}" for label, value in info]
            self.model_info.setText("\n".join(lines))
        except Exception as e:
            self.model_info.setText(f"加载失败: {e}")

    def _browse_output(self) -> None:
        """浏览选择输出目录。"""
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", str(self.output_dir_edit.text())
        )
        if path:
            self.output_dir_edit.setText(path)

    def get_params(self) -> dict:
        """获取当前参数。"""
        return {
            "model_key": self.model_select.currentData(),
            "device": "auto",
            "dtype": self.dtype_select.currentText(),
            "output_dir": self.output_dir_edit.text(),
        }
