# -*- coding: utf-8 -*-
"""预设管理（PLAN §2.12）。

预设是**参数的命名组合**，不替代参数面板。内置一个：
- `默认`：v2 模型 + float32 精度 + output/separated/ 输出目录

★按 §2.14，设备不是预设维度（GPU-only），所以预设结构里没有 device 字段。
旧预设如果带了 device 字段，加载时会被忽略而不是报错。

预设文件：~/.config/audio_separation/presets.json（Linux）
           %APPDATA%/audio_separation/presets.json（Windows）
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: 默认预设：v2 + float32 + 默认输出目录
DEFAULT_PRESETS = {
    "默认": {"model_key": "v2", "dtype": "float32", "output_dir": ""},
}

#: 预设字段白名单（§2.14 后设备不是可选维度，故意不存）
PRESET_FIELDS = {"model_key", "dtype", "output_dir"}


def presets_dir() -> Path:
    """预设配置目录。"""
    if os.name == "nt":
        # Windows: %APPDATA%\audio_separation\
        base = os.environ.get("APPDATA") or Path.home() / "AppData/Roaming"
    else:
        # Linux: ~/.config/audio_separation/
        base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    p = Path(base) / "audio_separation"
    p.mkdir(parents=True, exist_ok=True)
    return p


def presets_path() -> Path:
    """预设 JSON 文件路径。"""
    return presets_dir() / "presets.json"


def load_presets() -> dict[str, dict[str, Any]]:
    """加载预设。文件不存在则返回内置默认值。"""
    path = presets_path()
    if not path.is_file():
        return dict(DEFAULT_PRESETS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(DEFAULT_PRESETS)
        return data
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_PRESETS)


def save_presets(presets: dict[str, dict[str, Any]]) -> None:
    """保存预设到文件。"""
    path = presets_path()
    text = json.dumps(presets, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def apply_preset(preset: dict[str, Any]) -> dict[str, Any]:
    """应用预设到参数面板，返回参数 dict。

    只取 PRESET_FIELDS 白名单里的字段，其他忽略（兼容旧预设）。
    """
    return {
        "model_key": preset.get("model_key", "v2"),
        "dtype": preset.get("dtype", "float32"),
        "output_dir": preset.get("output_dir", ""),
    }


def save_preset(name: str, params: dict[str, Any]) -> None:
    """保存当前参数为一个新预设（或覆盖同名预设）。"""
    presets = load_presets()
    # 只存白名单字段
    clean = {k: params[k] for k in PRESET_FIELDS if k in params}
    presets[name] = clean
    save_presets(presets)


def delete_preset(name: str) -> None:
    """删除一个预设（内置的不能删）。"""
    if name in DEFAULT_PRESETS:
        raise ValueError(f"内置预设「{name}」不能被删除")
    presets = load_presets()
    presets.pop(name, None)
    save_presets(presets)
