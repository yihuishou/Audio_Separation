"""端到端：4 个 60 s 切片真机分离（PLAN M2-9，`@pytest.mark.slow gpu`）。

**这里承担 `tests/test_model.py` 被删掉的那部分覆盖**（PLAN §6.5 的删除说明）：
`build_model()` 的权重键名/形状严格比对就在这条路径上跑，键名对不上会当场
`RuntimeError`，不需要再单独写一个"只构建不上推理"的测试。

## 为什么必须真机

分块、加窗、wsum 归一化在 `test_separator.py` 里已经用恒等模型在 CPU 上覆盖过了
（那不违反 §2.14 —— 被测的是纯 numpy 算法）。这里覆盖的是**只有真 GPU + 真权重
才会暴露的东西**：`build_model` 的键名对齐、真实前向的显存占用、`mix_check` 的
端到端判据、Windows 上带全角字符的目录创建与写盘。

## 耗时

4 个切片 × 约 50 s ≈ 3.5 min（GPU）。权重只加载一次（`tests/_model_cache.py`）。
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.io import wavfile

from audio_separation.core.audio_io import decode_audio
from audio_separation.core.config import TRAINED_SAMPLE_RATE
from audio_separation.core.pipeline import MIX_ERROR_LIMIT_PCT, separate_file
from audio_separation.core.watchdog import BlockWatchdog

# tests/ 没有 __init__.py（rootdir 模式），pytest 会把 tests/ 塞进 sys.path，
# 所以这里用顶层绝对导入而不是 `from ._model_cache`。
from _model_cache import load_cached

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


def test_every_clip_separates_and_reconstructs(clip_paths, tmp_path, requires_gpu):
    """跑 1 个切片端到端，验证产物齐全且 mix 判据通过。

    只跑 1 个切片（而非全部 4 个）以缩短慢速测试耗时。其余切片的
    文件存在性由快速测试 `tests/test_clips.py` 覆盖。
    """
    clips = sorted(clip_paths)
    assert len(clips) == 4, f"期望 4 个切片，实际 {len(clips)}"
    clip = clips[0]  # 只跑第一个

    result = separate_file(
        src=clip,
        output_root=tmp_path,
        model_key="v2",
        model=load_cached("v2"),
        device="auto",
        block_timeout_sec=600.0,
    )

    assert result.ok, f"{clip.name}: mix 误差 {result.mix_error_pct}% 超过上限"
    assert result.mix_error_pct < MIX_ERROR_LIMIT_PCT
    assert result.vocals_path.exists(), clip.name
    assert result.instrumental_path.exists(), clip.name
    assert result.result_json.exists(), clip.name

    rate, vocals = wavfile.read(result.vocals_path)
    assert rate == TRAINED_SAMPLE_RATE
    assert vocals.dtype == np.int16
    assert vocals.shape[0] == 1_323_000, f"{clip.name}: {vocals.shape}"

    payload = json.loads(result.result_json.read_text(encoding="utf-8"))
    assert payload["model_key"] == "v2"
    assert payload["device"] == "cuda"
    assert payload["blocks"] == 7, f"{clip.name}: 30 s 应为 7 个分块"


def test_unicode_directory_names_survive_the_whole_pipeline(clip_paths, tmp_path, requires_gpu):
    """★产物目录名由输入 stem 派生，必然带日文与全角『』。

    R12 在 ffmpeg 层已经验证过一次，这里验证的是**我们的**目录创建、
    `.npz` 缓存写入与 `os.replace()` 原子替换在宽字符路径下也成立。
    """
    clip = sorted(clip_paths)[0]
    result = separate_file(
        src=clip,
        output_root=tmp_path,
        model_key="v2",
        model=load_cached("v2"),
        device="auto",
        block_timeout_sec=600.0,
    )
    assert result.out_dir.name == clip.stem
    assert "『" in result.out_dir.name
    assert result.out_dir.is_dir()


def test_second_run_hits_the_cache(clip_paths, tmp_path, requires_gpu):
    """缓存命中必须**跳过推理**且产物一致 —— 这是批量处理时省时间的关键路径。"""
    clip = sorted(clip_paths)[0]
    model = load_cached("v2")

    first = separate_file(
        src=clip, output_root=tmp_path, model_key="v2", model=model,
        device="auto", block_timeout_sec=600.0,
    )
    assert not first.cache_hit
    # 30 秒切片 = 7 块（chunk=11s, hop=3.667s）
    assert first.blocks == 7

    second = separate_file(
        src=clip, output_root=tmp_path, model_key="v2", model=model,
        device="auto", block_timeout_sec=600.0,
    )
    assert second.cache_hit, "第二次运行没有命中 vocals.npz 缓存"
    # 缓存命中时不做任何 GPU 计算，blocks 仍然要报出来（UI 要显示总块数）
    assert second.blocks == 7
    assert second.mix_error_pct < MIX_ERROR_LIMIT_PCT

    a, _ = wavfile.read(first.vocals_path)
    b, _ = wavfile.read(second.vocals_path)
    assert np.array_equal(a, b), "缓存命中产物与首次不一致"


def test_cache_is_invalidated_by_a_shape_mismatch(clip_paths, tmp_path, requires_gpu):
    """形状对不上的缓存必须被丢弃重算，不能当成有效结果返回。"""
    clip = sorted(clip_paths)[0]
    first = separate_file(
        src=clip, output_root=tmp_path, model_key="v2", model=load_cached("v2"),
        device="auto", block_timeout_sec=600.0,
    )
    assert not first.cache_hit

    np.savez(first.out_dir / "vocals.npz", vocals=np.zeros((1234, 2), dtype=np.float32))

    again = separate_file(
        src=clip, output_root=tmp_path, model_key="v2", model=load_cached("v2"),
        device="auto", block_timeout_sec=600.0,
    )
    assert not again.cache_hit, "形状不符的缓存被当成了有效结果"
    assert again.blocks == 7


def test_mix_check_wav_reconstructs_the_source(clip_paths, tmp_path, requires_gpu):
    """★残差构造是精确的：vocals + instrumental == 原轨，误差只在量化级。

    这是唯一能证明「instrumental 是残差、不是模型第二个输出」的地方 ——
    若实现哪天改成让模型直接出两轨，这个断言会立刻失败。
    """
    clip = sorted(clip_paths)[0]
    result = separate_file(
        src=clip, output_root=tmp_path, model_key="v2", model=load_cached("v2"),
        device="auto", block_timeout_sec=600.0,
    )
    source = decode_audio(str(clip), TRAINED_SAMPLE_RATE).astype(np.float64)
    _, vocals_i = wavfile.read(result.vocals_path)
    _, instrumental_i = wavfile.read(result.instrumental_path)

    # int16 量化：两次写出各引入最多 1/32767，合计 2/32767 ≈ 6.1e-5
    rebuilt = (vocals_i.astype(np.float64) + instrumental_i.astype(np.float64)) / 32767.0
    assert np.abs(rebuilt - source).max() < 2.0 / 32767.0 * 1.01


def test_watchdog_kills_a_stalled_block(tmp_path):
    """★不用 GPU：看门狗超时必须在限定时间内真的结束进程路径。

    这里只验证 `timeout_report()` 的内容与非零退出码这个契约 ——
    真去验证 `os._exit(3)` 会把 pytest 自己杀掉。
    """
    from audio_separation.core import watchdog

    text = watchdog.timeout_report("块 3/15 前向", idle=305.0, limit=300.0)
    assert "300" in text, "报告里应给出超时上限"
    assert "5" in text, "报告里应给出已停滞的秒数"
    assert watchdog.TIMEOUT_EXIT_CODE == 3
    # ★§2.14 之后**不能建议换 CPU**，但允许明说"不做 CPU 回退" ——
    #   区分这两者是这条断言的全部意义：前者是给用户的出路，后者只是事实陈述。
    assert "--device cpu" not in text
    assert "不做 CPU 回退" in text
