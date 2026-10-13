# -*- coding: utf-8 -*-
"""配置加载与交叉校验的测试（PLAN §2.1）。

PLAN 要求「测试取 v2/v3 两个 yaml 各断言一次」：未来换模型版本若导致
``chunk_size`` 的 −1 关系变化，这里会立刻红，而不是让用户在成品里
听出差别却找不到原因。
"""

import copy

import pytest
import yaml

from audio_separation import paths
from audio_separation.core.config import (
    TRAINED_SAMPLE_RATE,
    ConfigError,
    _YamlLoader,
    describe_config,
    load_config,
    load_config_by_key,
    validate_config,
)
from audio_separation.core.requirements import REQUIREMENTS

MODEL_KEYS = [info.key for info in paths.list_models()]


def _load_any():
    """本机只要有一对模型就能跑；素材不在仓库里时跳过而不是失败。"""
    if not MODEL_KEYS:
        pytest.skip("models/bs_roformer_revive/ 下没有成对的 yaml+ckpt")
    return load_config_by_key(MODEL_KEYS[0])


class TestPythonTupleTag:
    def test_python_tuple_tag_is_understood(self, tmp_path):
        """``!!python/tuple`` 必须还原成 tuple。

        PyTorch 导出的配置都带这个标签，SafeLoader 默认不认识，
        直接 load 会抛 ConstructorError —— 整个模型都读不出来。
        """
        raw = tmp_path / "t.yaml"
        raw.write_text(
            "model:\n"
            "  freqs_per_bands: !!python/tuple [2, 4, 129]\n"
            "  multi_stft_resolutions_window_sizes: !!python/tuple [4096, 2048]\n"
            "  stereo: true\n",
            encoding="utf-8",
        )
        cfg = yaml.load(raw.read_text(encoding="utf-8"), Loader=_YamlLoader)
        assert isinstance(cfg["model"]["freqs_per_bands"], tuple)
        assert isinstance(cfg["model"]["multi_stft_resolutions_window_sizes"], tuple)
        assert cfg["model"]["freqs_per_bands"] == (2, 4, 129)

    def test_real_config_has_tuple_fields(self):
        """真实的 v2/v3 yaml 里这两个字段确实是 tuple，模型构造时依赖这一点。"""
        cfg = _load_any()
        assert isinstance(cfg["model"]["freqs_per_bands"], tuple)
        assert isinstance(cfg["model"]["multi_stft_resolutions_window_sizes"], tuple)


class TestLoadConfig:
    def test_injects_dir_and_weights(self):
        cfg = _load_any()
        assert cfg["_weights"].endswith(".ckpt")
        assert cfg["_dir"] in cfg["_weights"]

    @pytest.mark.parametrize("key", MODEL_KEYS)
    def test_every_available_model_loads(self, key):
        """仓库里有几个模型就必须都能加载，不允许只支持其中一个。"""
        cfg = load_config_by_key(key)
        assert cfg["audio"]["sample_rate"] == TRAINED_SAMPLE_RATE

    def test_unknown_key_raises_with_available_list(self):
        with pytest.raises(ConfigError) as excinfo:
            load_config_by_key("v99")
        assert "v99" in str(excinfo.value)
        for key in MODEL_KEYS:
            assert key in str(excinfo.value), "报错应列出现有模型键，别让人猜"

    def test_missing_ckpt_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="权重文件不存在"):
            load_config(tmp_path / "nope.ckpt")

    def test_missing_yaml_raises(self, tmp_path):
        ckpt = tmp_path / "only.ckpt"
        ckpt.write_bytes(b"")
        with pytest.raises(ConfigError, match="没有配套的 yaml"):
            load_config(ckpt)

    def test_malformed_yaml_raises(self, tmp_path):
        ckpt = tmp_path / "bad.ckpt"
        ckpt.write_bytes(b"")
        (tmp_path / "bad.yaml").write_text("model: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="yaml 解析失败"):
            load_config(ckpt)

    def test_non_dict_top_level_raises(self, tmp_path):
        ckpt = tmp_path / "seq.ckpt"
        ckpt.write_bytes(b"")
        (tmp_path / "seq.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="顶层不是字典"):
            load_config(ckpt)


class TestCrossValidation:
    """§2.1 的四条交叉校验，逐条触发一次。"""

    def test_valid_config_has_no_problems(self):
        assert validate_config(_load_any()) == []

    def test_missing_section_reports_and_stops(self):
        problems = validate_config({"audio": {}, "model": {}, "training": {}})
        assert any("inference" in p for p in problems)

    def test_dim_freqs_in_mismatch_is_caught(self):
        cfg = copy.deepcopy(_load_any())
        cfg["model"]["dim_freqs_in"] += 1
        problems = validate_config(cfg)
        assert any("dim_freqs_in" in p for p in problems)
        assert any("频点数错位" in p for p in problems), "报错要说明后果，只说数值不符等于让人自己推"

    def test_wrong_sample_rate_is_caught(self):
        cfg = copy.deepcopy(_load_any())
        cfg["audio"]["sample_rate"] = 48000
        problems = validate_config(cfg)
        assert any("sample_rate" in p and "48000" in p for p in problems)

    def test_dim_t_mismatch_between_audio_and_inference_is_caught(self):
        cfg = copy.deepcopy(_load_any())
        cfg["inference"]["dim_t"] += 2
        problems = validate_config(cfg)
        assert any("inference.dim_t" in p for p in problems)

    def test_chunk_size_uses_dim_t_minus_one(self):
        """★核心：必须认 ``hop × (dim_t−1)``，且报错里点破写成 ``× dim_t`` 的错误值。"""
        cfg = copy.deepcopy(_load_any())
        wrong = cfg["audio"]["hop_length"] * cfg["audio"]["dim_t"]
        cfg["audio"]["chunk_size"] = wrong
        problems = validate_config(cfg)
        chunk_problems = [p for p in problems if "chunk_size" in p]
        assert chunk_problems, "chunk_size 写错却没有被拦下"
        assert str(wrong) in chunk_problems[0], "报错应直接给出 ×dim_t 会算出的错误值"

    def test_all_problems_reported_at_once(self):
        """★一次报全部问题，别让人改一遍跑一次。"""
        cfg = copy.deepcopy(_load_any())
        cfg["audio"]["sample_rate"] = 48000
        cfg["model"]["dim_freqs_in"] += 3
        cfg["inference"]["dim_t"] += 1
        cfg["audio"]["chunk_size"] += 7
        problems = validate_config(cfg)
        assert len(problems) == 4, f"应有 4 条问题，实际 {len(problems)}：{problems}"


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_chunk_size_relation_holds_for_real_models(key):
    """★PLAN 明确要求：v2/v3 各断言一次「chunk_size == hop × (dim_t − 1)」。

    这是那个「写错一个字符、模型静默输出错位人声」的关系。
    """
    audio = load_config_by_key(key)["audio"]
    assert audio["chunk_size"] == audio["hop_length"] * (audio["dim_t"] - 1)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_freqs_per_bands_sums_to_dim_freqs_in(key):
    """★频带划分之和必须等于模型吃进去的频点数，否则 STFT 输出对不上。"""
    model = load_config_by_key(key)["model"]
    assert sum(model["freqs_per_bands"]) == model["dim_freqs_in"]


class TestDescribeConfig:
    def test_returns_label_value_pairs(self):
        rows = describe_config(_load_any())
        assert rows and all(len(row) == 2 for row in rows)
        assert all(isinstance(label, str) and isinstance(value, str) for label, value in rows)

    def test_includes_the_four_locked_constants(self):
        """§2.11 面板要展示的正是这几项，漏一项就等于没展示。"""
        labels = [label for label, _ in describe_config(_load_any())]
        blob = " ".join(labels)
        for keyword in ("采样率", "分块长度", "时间维", "重叠块数"):
            assert keyword in blob, f"只读面板缺少 {keyword}"


class TestRequirementsStillAlign:
    def test_declared_min_versions_match_installed_environment(self):
        """§2.13 的 min_version 是从本机实测抄来的，装完环境后必须还成立。"""
        import importlib.metadata

        for entry in REQUIREMENTS:
            if entry.get("is_python") or entry.get("is_embedded") or entry.get("group") == "G":
                continue
            try:
                installed = importlib.metadata.version(entry["name"])
            except importlib.metadata.PackageNotFoundError:
                continue
            from audio_separation.core.env_check import _below

            assert not _below(installed, entry["min_version"]), (
                f"{entry['name']} 实装 {installed}，低于声明的 {entry['min_version']}；"
                "§2.13 的清单需要更新"
            )
