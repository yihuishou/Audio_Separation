# -*- coding: utf-8 -*-
"""测试素材（test_audio/clips/）的形态校验。

切片是本项目的性能与正确性基线：全曲 513 MB / 32 分钟绝不能进测试，
所以 ``tools/make_test_clips.py`` 从两个源文件各切两段 60 s。

★60 s 不是随手选的：``chunk_size = 485100`` ≈ 11.0 s，
60 s 恰好跨 14 个分块、含 13 个内部重叠边界；
若只用 10 s 就只有一块，overlap 加权与 wsum 归一化一行代码都覆盖不到。

整组标记为 slow：素材缺失时会先触发生成，最长可能跑一分多钟。
"""

import wave

import pytest

from audio_separation import paths as audio_paths

#: 每首切两段，起点 240 s 与 600 s（600 s 那段接近文件尾部）
EXPECTED_CLIP_COUNT = 4
EXPECTED_DURATION_S = 30.0
EXPECTED_SAMPLE_RATE = 44100
EXPECTED_CHANNELS = 2
EXPECTED_OFFSETS = ("t240s", "t600s")


@pytest.mark.slow
def test_clips_exist_and_count(clips):
    assert len(clips) == EXPECTED_CLIP_COUNT, f"应有 {EXPECTED_CLIP_COUNT} 个切片，实际 {len(clips)}"


@pytest.mark.slow
def test_clips_are_nonempty(clips):
    for clip in clips:
        assert clip.stat().st_size > 0, f"切片 {clip.name} 是空文件"


@pytest.mark.slow
def test_clips_are_readable_wav(clips):
    for clip in clips:
        with wave.open(str(clip), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
        assert channels == EXPECTED_CHANNELS, f"{clip.name} 是 {channels} 声道，应为 {EXPECTED_CHANNELS}"
        assert sample_rate == EXPECTED_SAMPLE_RATE, f"{clip.name} 是 {sample_rate} Hz，应为 {EXPECTED_SAMPLE_RATE}"
        assert frames > 0, f"{clip.name} 帧数为 0"


@pytest.mark.slow
def test_clips_have_exact_duration(clips):
    """时长必须正好 60.000 s。

    早先这里写的是 ``duration in (60.0, 240.0)``——把「起点偏移 240 s」
    误当成了「允许的时长」，于是任何多出来的秒数都能蒙混过关。
    """
    for clip in clips:
        with wave.open(str(clip), "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
        duration = frames / sample_rate
        assert duration == EXPECTED_DURATION_S, (
            f"{clip.name} 时长 {duration:.3f}s，应为 {EXPECTED_DURATION_S}s"
        )


@pytest.mark.slow
def test_clip_frame_count_is_exact(clips):
    """帧数必须是 44100×60，不允许因为切分取整少一帧。

    少一帧会让后面所有逐样本比对（mix 校验、回归基线）在末尾错开。
    """
    for clip in clips:
        with wave.open(str(clip), "rb") as wf:
            frames = wf.getnframes()
        assert frames == EXPECTED_SAMPLE_RATE * EXPECTED_DURATION_S, (
            f"{clip.name} 有 {frames} 帧，应为 {EXPECTED_SAMPLE_RATE * EXPECTED_DURATION_S}"
        )


@pytest.mark.slow
def test_clips_cover_both_sources_and_both_offsets(clips):
    """4 个切片必须来自 2 个源文件 × 2 个起点，缺一说明生成逻辑漏了分支。"""
    stems = {clip.stem for clip in clips}
    for offset in EXPECTED_OFFSETS:
        matched = [s for s in stems if s.endswith(f"_{offset}")]
        assert len(matched) == 2, f"起点 {offset} 应各有一个切片（来自 2 首歌），实际 {matched}"

    prefixes = {clip.stem.rsplit("_", 1)[0] for clip in clips}
    assert len(prefixes) == 2, f"应来自 2 个不同的源文件，实际 {prefixes}"


@pytest.mark.slow
def test_clip_filenames_preserve_source_unicode(clips):
    """★切片名必须原样保留源文件里的日文与全角『』。

    这是对 ffmpeg 在 Windows 上按宽字符解析 argv 的实打实检验
    （R12）；一旦走了「拷到纯 ASCII 临时名」的兜底方案，这里就会红。
    """
    for clip in clips:
        assert any(ord(ch) > 127 for ch in clip.stem), (
            f"{clip.name} 的名字里没有非 ASCII 字符，源文件名的日文被丢掉了"
        )
    assert any("『" in clip.stem and "』" in clip.stem for clip in clips), (
        "没有切片保留全角书名号『』，Unicode 传递链路可能已退化"
    )


def test_clips_dir_is_under_test_audio(repo_root):
    """切片目录挂在 test_audio/ 下（纯路径关系检查，不需要素材存在）。"""
    clips_dir = audio_paths.test_clips_dir()
    assert clips_dir.resolve().is_relative_to(repo_root.resolve())
    assert clips_dir.parent.name == "test_audio"
