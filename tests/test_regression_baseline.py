"""M6 回归基线：验证分离结果是确定性的（同一输入两次运行输出相同）。

真正的回归基线需要历史产物，但本项目是新实现，没有"之前"。
所以这里的策略是：
1. 对同一输入跑两次，比较输出是否相同（确定性）
2. 验证重建精度（vocals + instrumental == original，容差内）
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from audio_separation.core.config import TRAINED_SAMPLE_RATE
from audio_separation.core.pipeline import separate_file

# tests/ 没有 __init__.py（rootdir 模式），pytest 会把 tests/ 塞进 sys.path，
# 所以这里用顶层绝对导入而不是 `from ._model_cache`。
from _model_cache import load_cached


@pytest.mark.slow
@pytest.mark.gpu
class TestRegressionBaseline:
    """分离结果的确定性和重建精度。"""

    @pytest.fixture(scope="class")
    def model(self):
        """共享模型，避免重复加载（610 MB）。"""
        return load_cached("v2")

    @pytest.fixture(scope="class")
    def first_run(self, clips, tmp_path_factory, model):
        """第一次运行：分离第一个切片。"""
        if not clips:
            pytest.skip("no test clips available")
        clip = clips[0]
        out_dir = tmp_path_factory.mktemp("baseline_run1")
        result = separate_file(
            clip,
            output_root=out_dir,
            model_key="v2",
            model=model,
            device="auto",
            dtype="float32",
            use_cache=False,
        )
        return result

    @pytest.fixture(scope="class")
    def second_run(self, clips, tmp_path_factory, model, first_run):
        """第二次运行：同一输入，验证确定性。"""
        clip = clips[0]
        out_dir = tmp_path_factory.mktemp("baseline_run2")
        result = separate_file(
            clip,
            output_root=out_dir,
            model_key="v2",
            model=model,
            device="auto",
            dtype="float32",
            use_cache=False,
        )
        return result

    def test_deterministic_output(self, first_run, second_run):
        """两次运行的 vocals 输出应相同（或非常接近，考虑到浮点误差）。"""
        sr1, v1 = wavfile.read(first_run.vocals_path)
        sr2, v2 = wavfile.read(second_run.vocals_path)

        assert sr1 == sr2
        assert v1.shape == v2.shape

        v1_f = v1.astype(np.float64)
        v2_f = v2.astype(np.float64)

        # 最大绝对差
        max_diff = np.abs(v1_f - v2_f).max()

        # 由于浮点运算和 CUDA 的非确定性，允许微小差异
        # int16 量化后，差异应该在量化步长以内
        assert max_diff <= 2, (
            f"两次运行的 vocals 差异过大：max_diff={max_diff}，"
            f"期望 <= 2（int16 量化步长）"
        )

    def test_reconstruction_precision(self, clips, first_run):
        """vocals + instrumental 应重建回原轨（容差内）。"""
        if not clips:
            pytest.skip("no test clips available")
        clip = clips[0]

        # 解码原轨为 float32
        from audio_separation.core.audio_io import decode_audio
        original = decode_audio(clip, TRAINED_SAMPLE_RATE)

        # 读取分离产物
        _, v = wavfile.read(first_run.vocals_path)
        _, i = wavfile.read(first_run.instrumental_path)

        # 转换为 float32（与解码后的原轨对齐）
        v_f = v.astype(np.float64) / 32767.0
        i_f = i.astype(np.float64) / 32767.0

        # 重建
        reconstructed = v_f + i_f

        # 计算差异
        diff = np.abs(reconstructed - original)
        max_diff = diff.max()
        rms_diff = np.sqrt((diff ** 2).mean())

        # int16 量化误差的理论上限是 2/32767 ≈ 6.1e-5
        # 允许一定余量
        assert max_diff < 1e-3, f"重建差异过大：max={max_diff}, rms={rms_diff}"

    def test_result_json_written(self, first_run):
        """result.json 必须存在且包含关键字段。"""
        assert first_run.result_json.exists()

        data = json.loads(first_run.result_json.read_text(encoding="utf-8"))
        required = ["src", "out_dir", "vocals", "instrumental",
                    "model_key", "mix_error_pct"]
        for field in required:
            assert field in data, f"result.json 缺少字段：{field}"
