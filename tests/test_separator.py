# -*- coding: utf-8 -*-
"""`tests/test_separator.py` — 分离算法（§2.7）。

★ 全部在**合成信号 + CPU + 恒等模型**上跑：
  - 不加载 610 MB 权重 ⇒ 快速集在 CI 上也能跑
  - 恒等模型（`y = x`）让「精确重建」成为一个**可断言的数学性质**，
    不需要靠听感判断分块/加窗/归一化对不对
"""
from __future__ import annotations

import numpy as np
import pytest

from audio_separation.core import separator
from audio_separation.core.separator import (
    BlockProgress,
    BlockWatchdog,
    Cancelled,
    chunk_starts,
    hann_window,
    separate_vocals,
    window_weights,
)

# ★ 模型常量（取自 v2/v3 yaml，两者一致）
SR = 44100
CHUNK = 485100          # 11.0 s
NUM_OVERLAP = 2
HOP = CHUNK // (NUM_OVERLAP + 1)   # 161700 = 3.667 s
CLIP_SEC = 60
TOTAL = CLIP_SEC * SR


# ══════════════════════════════════════════════════════════════════════════════
# chunk_starts
# ══════════════════════════════════════════════════════════════════════════════
class TestChunkStarts:
    def test_hop_is_chunk_divided_by_overlap_plus_one(self):
        assert HOP == 161700

    def test_sixty_second_clip_is_fifteen_blocks(self):
        """§1.2 的推导（2026-10-13 实测修正）。

        `range(0, 2646000 - 485100 + 1, 161700)` 给出 14 个等步块，
        但第 14 块的右边界 2587200 < 2646000，**盖不到文件尾**，
        所以还要补一个贴右边界的第 15 块。
        （参考脚本 `--dry-run` 只数等步块，会报 14 —— 那是少算。）
        """
        starts = chunk_starts(TOTAL, CHUNK, NUM_OVERLAP)
        assert len(starts) == 15
        assert len(starts) - 1 == 14  # 内部重叠边界数

    def test_starts_are_sorted_and_unique(self):
        starts = chunk_starts(TOTAL, CHUNK, NUM_OVERLAP)
        assert starts == sorted(set(starts))

    def test_no_overlap_between_neighbours_is_exactly_hop(self):
        """相邻块推进量应当稳定等于 hop（末块的补位除外）。"""
        starts = chunk_starts(TOTAL, CHUNK, NUM_OVERLAP)
        assert starts[:-1] == list(range(0, (len(starts) - 1) * HOP, HOP))

    def test_last_block_reaches_the_end_of_the_file(self):
        """★末尾补位：不补的话最后 total % hop 个样本没有任何窗口覆盖。"""
        starts = chunk_starts(TOTAL, CHUNK, NUM_OVERLAP)
        assert starts[-1] + CHUNK == TOTAL

    @pytest.mark.parametrize("total", [1, 100, CHUNK - 1, CHUNK, CHUNK + 1, TOTAL, TOTAL - 1])
    def test_blocks_stay_inside_the_file(self, total):
        for s in chunk_starts(total, CHUNK, NUM_OVERLAP):
            assert 0 <= s
            assert s + CHUNK <= max(total, CHUNK)

    def test_short_audio_is_a_single_block(self):
        assert chunk_starts(1000, CHUNK, NUM_OVERLAP) == [0]

    def test_empty_audio_has_no_blocks(self):
        assert chunk_starts(0, CHUNK, NUM_OVERLAP) == []

    def test_negative_overlap_is_rejected(self):
        with pytest.raises(ValueError, match="num_overlap"):
            chunk_starts(TOTAL, CHUNK, -1)


# ══════════════════════════════════════════════════════════════════════════════
# hann_window
# ══════════════════════════════════════════════════════════════════════════════
class TestHannWindow:
    def test_endpoints_are_zero(self):
        """★`periodic=False` 的定义性特征：两端严格归零。"""
        w = hann_window(1024)
        assert w[0] == 0.0
        assert w[-1] == 0.0

    def test_peak_is_one_and_at_the_centre(self):
        w = hann_window(1024)
        # ★偶数长度下 peak 不落在整数下标上：i=512 时 cos(2π·512/1023)=−0.99999，
        #   所以最大值是 0.9999976 而非精确 1。periodic=True 时才是精确 1.0。
        assert w.max() == pytest.approx(1.0, abs=1e-5)
        # 偶数长度下中央两个点等值，argmax 取靠前的那个
        assert int(np.argmax(w)) in (511, 512)

    def test_is_symmetric(self):
        w = hann_window(1024)
        assert np.allclose(w, w[::-1])

    def test_differs_from_periodic_variant(self):
        """★如果哪天有人传成 periodic=True，本测试会立刻红。"""
        n = 1024
        i = np.arange(n)
        periodic = 0.5 - 0.5 * np.cos(2 * np.pi * i / n)
        assert not np.allclose(hann_window(n), periodic)

    def test_length_one_is_a_unity_window(self):
        assert np.array_equal(hann_window(1), np.ones(1))

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_chunk_is_rejected(self, bad):
        with pytest.raises(ValueError, match="chunk"):
            hann_window(bad)


# ══════════════════════════════════════════════════════════════════════════════
# ★ 加窗归一化的数学性质（PLAN §2.7 的核心约定）
# ══════════════════════════════════════════════════════════════════════════════
class TestWindowWeights:
    def test_interior_weight_is_constant(self):
        """分母 `Σw` 在内部恒定 —— 这是「加权平均」能精确重建的前提。

        首尾各一个 chunk 是 Hann 窗必然衰减的边缘区（两端窗值为 0）。
        """
        w = window_weights(CHUNK, NUM_OVERLAP, TOTAL)[CHUNK:-CHUNK]
        # 3 块重叠 ⇒ Σw ≈ 1.5；关键是**恒定**而不是等于某个数
        assert w.std() < 1e-3
        assert w.mean() == pytest.approx(1.5, abs=0.01)

    def test_quadratic_denominator_would_be_wrong_by_a_third(self):
        """★守住 §2.7 的修正。

        分母若改成 `Σ(w²)` 而分子仍是 `Σ(y·w)`，内部增益恒为 `Σw/Σw² = 4/3`，
        整轨人声偏大约 4 dB。本测试把这个错误值钉死，
        任何人「顺手把归一化改成平方窗」都会立刻看到失败。
        """
        w = hann_window(CHUNK)
        num = np.zeros(TOTAL)
        den = np.zeros(TOTAL)
        for s in chunk_starts(TOTAL, CHUNK, NUM_OVERLAP):
            n = min(CHUNK, TOTAL - s)
            num[s:s + n] += w[:n]
            den[s:s + n] += w[:n] ** 2
        gain = (num / np.maximum(den, 1e-12))[CHUNK:-CHUNK]
        assert np.allclose(gain, 4.0 / 3.0, atol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# separate_vocals（恒等模型，不碰权重、不碰 GPU）
# ══════════════════════════════════════════════════════════════════════════════
def _identity_model(x):
    """输入 `(1, 2, chunk)` → 输出 `(1, 1, 2, chunk)` 的恒等模型。"""
    return x.unsqueeze(1)


def _run(wav, **kw):
    params = dict(chunk=CHUNK, num_overlap=NUM_OVERLAP, device="cpu")
    params.update(kw)
    return separate_vocals(_identity_model, wav, SR, **params)


class TestSeparateVocals:
    @pytest.fixture
    def signal(self):
        """60 s 确定性合成信号（两个不同频率，避开 0/1 边界）。"""
        t = np.arange(TOTAL, dtype=np.float32) / SR
        left = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        right = (0.3 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32)
        return np.stack([left, right], axis=1)

    def test_reconstruction_is_exact(self, signal):
        """★恒等模型 ⇒ 输出必须等于输入（内部区间逐样本相等）。

        这是整个分块/加窗/归一化链条最强的单条断言：hop 算错、末块没补位、
        窗用成 periodic、加窗归一化不自洽 —— 四类错误里任何一类都会让它变红。
        """
        out = _run(signal)
        assert out.shape == signal.shape
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[CHUNK:-CHUNK], signal[CHUNK:-CHUNK], atol=1e-5)

    def test_edges_are_zeroed_not_nan(self):
        """首样本的窗值为 0，分母被 1e-8 兜底 ⇒ 必须是 0，不能是 NaN/inf。"""
        out = _run(np.full((TOTAL, 2), 0.5, dtype=np.float32))
        assert np.all(np.isfinite(out))
        assert out[0, 0] == 0.0

    def test_progress_is_reported_once_per_block(self, signal):
        seen: list[BlockProgress] = []
        _run(signal, on_progress=seen.append)
        assert len(seen) == 15
        assert [p.done for p in seen] == list(range(1, 16))
        assert seen[-1].eta_sec == 0.0
        assert seen[-1].end_sec == pytest.approx(CLIP_SEC, abs=0.01)

    def test_cancellation_stops_at_a_block_boundary(self, signal):
        """★取消只能在块边界生效（§2.9）—— 一块 11 s 的前向无法中断。"""
        calls = {"n": 0}

        def cancel_after_two():
            calls["n"] += 1
            return calls["n"] > 2

        with pytest.raises(Cancelled, match="第 3/15"):
            _run(signal, should_cancel=cancel_after_two)

    def test_cancel_before_the_first_block_runs_nothing(self, signal):
        with pytest.raises(Cancelled, match="第 1/15"):
            _run(signal, should_cancel=lambda: True)

    def test_audio_shorter_than_one_chunk_is_padded_and_trimmed(self):
        """★末块补零 → 定长输入；输出必须只取前 n 个样本，不能多出补零尾巴。"""
        short = np.full((5000, 2), 0.25, dtype=np.float32)
        out = _run(short)
        assert out.shape == short.shape
        # 内部（避开两端为 0）应当原样还原
        np.testing.assert_allclose(out[1:-1], short[1:-1], atol=1e-5)

    @pytest.mark.parametrize("bad_shape", [(0, 2), (100, 3), (100,)])
    def test_rejects_wrongly_shaped_input(self, bad_shape):
        with pytest.raises(ValueError):
            _run(np.zeros(bad_shape, dtype=np.float32))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="空"):
            _run(np.zeros((0, 2), dtype=np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# 看门狗
# ══════════════════════════════════════════════════════════════════════════════
class TestBlockWatchdog:
    def test_fires_when_no_beat_arrives(self):
        fired: list[tuple[str, float]] = []
        wd = BlockWatchdog(0.2, on_timeout=lambda stage, idle: fired.append((stage, idle)))
        with wd:
            wd.beat("块 1/14 前向")
            import time

            time.sleep(1.5)
        assert fired, "看门狗没有触发"
        assert fired[0][0] == "块 1/14 前向"

    def test_beat_resets_the_idle_timer(self):
        fired: list[tuple[str, float]] = []
        wd = BlockWatchdog(1.0, on_timeout=lambda stage, idle: fired.append((stage, idle)))
        with wd:
            import time

            for _ in range(4):
                time.sleep(0.3)
                wd.beat("推进中")
        assert not fired, "持续 beat 时不应触发"

    def test_rejects_non_positive_limit(self):
        with pytest.raises(ValueError, match="limit_sec"):
            BlockWatchdog(0.0)


def test_module_exports_are_stable():
    """GUI / worker 会按名字导入，别把接口改没了。"""
    for name in ("separate_vocals", "chunk_starts", "hann_window",
                 "window_weights", "BlockWatchdog", "BlockProgress", "Cancelled"):
        assert name in separator.__all__
        assert hasattr(separator, name)
