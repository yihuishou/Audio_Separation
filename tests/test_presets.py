# -*- coding: utf-8 -*-
"""预设管理测试（PLAN §2.12 / §4 M5）。"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from audio_separation.core.presets import (
    DEFAULT_PRESETS,
    PRESET_FIELDS,
    apply_preset,
    delete_preset,
    load_presets,
    presets_path,
    save_preset,
    save_presets,
)


@pytest.fixture
def tmp_config_dir(tmp_path: Path, monkeypatch) -> Path:
    """临时配置目录，测试完自动清理。"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    if os.name == "nt":
        monkeypatch.setenv("APPDATA", str(config_dir))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
    return config_dir


def test_load_returns_defaults_when_file_missing(tmp_config_dir: Path) -> None:
    """预设文件不存在时返回内置默认值。"""
    presets = load_presets()
    assert presets == DEFAULT_PRESETS


def test_load_existing_file(tmp_config_dir: Path) -> None:
    """能从文件加载预设。"""
    custom = {"我的预设": {"model_key": "v3", "dtype": "float16", "output_dir": "/tmp/out"}}
    path = presets_path()
    path.write_text(json.dumps(custom), encoding="utf-8")

    presets = load_presets()
    assert presets == custom


def test_load_corrupted_file(tmp_config_dir: Path) -> None:
    """文件损坏时返回默认值。"""
    path = presets_path()
    path.write_text("not valid json {", encoding="utf-8")

    presets = load_presets()
    assert presets == DEFAULT_PRESETS


def test_save_and_load_roundtrip(tmp_config_dir: Path) -> None:
    """保存后能正确加载。"""
    presets = {
        "测试预设": {"model_key": "v2", "dtype": "float32", "output_dir": "/tmp"},
    }
    save_presets(presets)

    loaded = load_presets()
    assert loaded == presets


def test_save_preset_overwrites(tmp_config_dir: Path) -> None:
    """保存同名预设时覆盖旧值。"""
    save_preset("测试", {"model_key": "v2", "dtype": "float32", "output_dir": "/tmp"})
    save_preset("测试", {"model_key": "v3", "dtype": "float16", "output_dir": "/tmp2"})

    presets = load_presets()
    assert presets["测试"]["model_key"] == "v3"


def test_save_preset_stores_only_whitelisted_fields(tmp_config_dir: Path) -> None:
    """只存白名单字段，其他忽略。"""
    save_preset("测试", {
        "model_key": "v2",
        "dtype": "float32",
        "output_dir": "/tmp",
        "device": "cuda",  # 白名单外，应被忽略
        "chunk_size": 485100,  # 白名单外，应被忽略
    })

    presets = load_presets()
    stored = presets["测试"]
    assert set(stored.keys()) == PRESET_FIELDS


def test_load_preset_with_extra_fields(tmp_config_dir: Path) -> None:
    """加载带多余字段的旧预设不报错。"""
    old_preset = {"我的预设": {"model_key": "v2", "dtype": "float32", "output_dir": "/tmp", "device": "auto"}}
    path = presets_path()
    path.write_text(json.dumps(old_preset), encoding="utf-8")

    presets = load_presets()
    params = apply_preset(presets["我的预设"])
    assert params["model_key"] == "v2"
    assert params["dtype"] == "float32"
    assert params["output_dir"] == "/tmp"


def test_apply_preset_returns_correct_fields() -> None:
    """apply_preset 返回正确的参数字段。"""
    preset = {"model_key": "v3", "dtype": "float16", "output_dir": "/custom"}
    params = apply_preset(preset)
    assert params["model_key"] == "v3"
    assert params["dtype"] == "float16"
    assert params["output_dir"] == "/custom"


def test_apply_preset_defaults() -> None:
    """预设字段缺失时使用默认值。"""
    params = apply_preset({})
    assert params["model_key"] == "v2"
    assert params["dtype"] == "float32"
    assert params["output_dir"] == ""


def test_delete_preset(tmp_config_dir: Path) -> None:
    """删除用户预设成功。"""
    save_preset("测试", {"model_key": "v2", "dtype": "float32", "output_dir": "/tmp"})
    delete_preset("测试")

    presets = load_presets()
    assert "测试" not in presets


def test_delete_builtin_preset_fails(tmp_config_dir: Path) -> None:
    """删除内置预设报错。"""
    with pytest.raises(ValueError, match="内置预设"):
        delete_preset("默认")
