# -*- coding: utf-8 -*-
"""`core.audio_io` 的测试（§2.4）。

判据分层：
- 纯 numpy 逻辑（量化、截断、RMS、mix 校验）→ 在临时目录里跑，毫秒级；
- 真正调用 ffmpeg 的解码测试 → 用 `pytest.importorskip` 之外的显式守卫，
  ffmpeg 缺失就 skip 而不是 fail（ffmpeg 是内嵌资源，不是可选项，
  但打包事故由 `test_packaging.py` 负责报错，不该让快速集变红）。

★ Unicode 往返是这里最要紧的一条：输入名带日文与全角『』，
  产物名由 stem 派生，必然继续带这些字符。Windows 上如果 ffmpeg 按窄字符
  解析 argv，整条链路会在第一步就断掉（R12）。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from audio_separation import paths
from audio_separation.core import audio_io
from audio_separation.core.audio_io import (
    AudioIOError,
    decode_audio,
    mix_error_pct,
    rms,
    write_wav,
)

SR = 44100

#: 故意用日文 + 全角『』，与 test_audio/ 的真实命名一致
UNICODE_NAME = "1.第1章『耳なめ手コキオナサポ』.wav"


@pytest.fixture
def stereo() -> np.ndarray:
    """2 秒确定性立体声：左右用不同频率，避免「双声道相同」的退化情况。"""
    t = np.arange(2 * SR, dtype=np.float64) / SR
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.5 * np.sin(2 * np.pi * 660.0 * t)
    return np.stack([left, right], axis=1).astype(np.float32)


def _ffmpeg_or_skip() -> Path:
    exe = paths.find_ff()
    if exe is None:
        pytest.skip(f"找不到 ffmpeg（期望 {paths.ffmpeg_bin()}）")
    return exe


# ══════════════════════════════════════════════════════════════════════════════
# write_wav
# ══════════════════════════════════════════════════════════════════════════════
class TestWriteWav:
    def test_roundtrip_preserves_shape_rate_and_pcm(self, tmp_path: Path, stereo: np.ndarray) -> None:
        import scipy.io.wavfile as sfw

        out = write_wav(tmp_path / "a.wav", stereo, SR)
        assert out == tmp_path / "a.wav"

        rate, pcm = sfw.read(str(out))
        assert rate == SR
        assert pcm.dtype == np.dtype("<i2")  # 16bit PCM，与 PLAN §2.4 一致
        assert pcm.shape == stereo.shape
        # 量化误差上限：1 个量化步长
        restored = pcm.astype(np.float64) / 32767.0
        assert np.abs(restored - stereo).max() < 1.0 / 32767.0

    def test_out_of_range_is_clipped_not_wrapped(self, tmp_path: Path) -> None:
        """越界值必须被截断。

        ★ 若用 32768 当系数，-1.0 会溢出成 +32767（正负翻转），
          波形变成巨大的方波 —— 这是真实会发生的错误，不是理论问题。
        """
        import scipy.io.wavfile as sfw

        loud = np.array([[-2.0, -1.5], [1.5, 2.0], [0.0, 0.0]], dtype=np.float32)
        out = write_wav(tmp_path / "loud.wav", loud, SR)
        _, pcm = sfw.read(str(out))
        assert pcm[:, 0].tolist() == [-32767, 32767, 0]
        assert pcm[:, 1].tolist() == [-32767, 32767, 0]

    def test_negative_full_scale_stays_negative(self, tmp_path: Path) -> None:
        """-1.0 必须写成 -32767 而不是 +32767。"""
        import scipy.io.wavfile as sfw

        out = write_wav(tmp_path / "x.wav", np.full((4, 2), -1.0, np.float32), SR)
        _, pcm = sfw.read(str(out))
        assert (pcm == -32767).all()

    def test_mono_is_accepted_and_written_as_mono(self, tmp_path: Path) -> None:
        import scipy.io.wavfile as sfw

        out = write_wav(tmp_path / "m.wav", np.zeros(100, np.float32), SR)
        _, pcm = sfw.read(str(out))
        assert pcm.shape == (100,)

    def test_no_temp_file_is_left_behind(self, tmp_path: Path, stereo: np.ndarray) -> None:
        """原子替换的另一半承诺：成功路径下 .tmp 已被 rename 掉。"""
        out = write_wav(tmp_path / "clean.wav", stereo, SR)
        assert not list(tmp_path.glob("*.tmp"))
        assert out.exists()

    def test_failure_does_not_leave_a_partial_file(self, tmp_path: Path, stereo: np.ndarray) -> None:
        """写到一半失败时，磁盘上既不能有目标文件也不能有 .tmp。"""
        target = tmp_path / "broken.wav"
        with pytest.raises(AudioIOError, match="写入失败"):
            _force_failure(tmp_path, stereo)
        assert not target.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_empty_array_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(AudioIOError, match="空数组"):
            write_wav(tmp_path / "e.wav", np.zeros((0, 2), np.float32), SR)

    def test_wrong_rank_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(AudioIOError, match="ndim"):
            write_wav(tmp_path / "d.wav", np.zeros((2, 2, 2), np.float32), SR)

    def test_unicode_filename_roundtrip(self, tmp_path: Path, stereo: np.ndarray) -> None:
        """产物名由输入 stem 派生，必然带日文与全角『』。"""
        import scipy.io.wavfile as sfw

        out = write_wav(tmp_path / UNICODE_NAME, stereo, SR)
        assert out.exists()
        assert out.name == UNICODE_NAME
        rate, _ = sfw.read(str(out))
        assert rate == SR

    def test_parent_directory_is_not_created(self, tmp_path: Path, stereo: np.ndarray) -> None:
        """§2.10：路径真源只解析不创建，缺失目录必须报错而不是静默 mkdir。"""
        with pytest.raises(AudioIOError, match="写入失败"):
            write_wav(tmp_path / "nope" / "x.wav", stereo, SR)
        assert not (tmp_path / "nope").exists()


def _force_failure(tmp_path: Path, stereo: np.ndarray) -> None:
    """把 `sfw.write` 替换成必抛 OSError 的桩，验证失败清理路径。"""
    import scipy.io.wavfile as sfw

    original = sfw.write

    def boom(path: str, *_args: object, **_kwargs: object) -> None:
        Path(path).write_bytes(b"partial")  # 模拟写了一半才炸
        raise OSError("磁盘满")

    sfw.write = boom  # type: ignore[assignment]
    try:
        write_wav(tmp_path / "broken.wav", stereo, SR)
    finally:
        sfw.write = original  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════════════════════════
# decode_audio
# ══════════════════════════════════════════════════════════════════════════════
class TestDecodeAudio:
    def test_missing_input_reports_the_path(self, tmp_path: Path) -> None:
        _ffmpeg_or_skip()
        with pytest.raises(AudioIOError, match="输入不存在"):
            decode_audio(tmp_path / "没有这个文件.wav", SR)

    def test_empty_input_is_reported(self, tmp_path: Path) -> None:
        _ffmpeg_or_skip()
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(AudioIOError, match="空文件"):
            decode_audio(empty, SR)

    def test_missing_ffmpeg_names_the_expected_location(self, tmp_path: Path) -> None:
        ghost = tmp_path / "ffmpeg.exe"
        with pytest.raises(AudioIOError) as excinfo:
            decode_audio(tmp_path / "x.wav", SR, ffmpeg=ghost)
        text = str(excinfo.value)
        assert "指定的 ffmpeg 不存在" in text
        # 必须指向内嵌目录，否则用户会去系统 PATH 里翻
        assert "ffmpeg" in text and paths.repo_root() is not None

    def test_unicode_source_is_decoded(self, tmp_path: Path, stereo: np.ndarray) -> None:
        """★ 最关键的一条：ffmpeg 在 Windows 上按宽字符解析 argv。"""
        _ffmpeg_or_skip()
        src = write_wav(tmp_path / UNICODE_NAME, stereo, SR)
        wav = decode_audio(src, SR)
        assert wav.shape == stereo.shape
        assert wav.dtype == np.float32
        # 容差 2 LSB：写盘向零截断 ≤1 LSB，ffmpeg 读回时按 /32768 缩放又差 1 LSB
        assert np.abs(wav - stereo).max() < 2.0 / 32767.0

    def test_unicode_output_name_survives_a_real_transcode(self, tmp_path: Path, stereo: np.ndarray) -> None:
        """整条链：Unicode 输入 → pcm_s24le → ffmpeg 解码。"""
        _ffmpeg_or_skip()
        pcm24 = tmp_path / "源.wav"
        write_wav(pcm24, stereo, SR)
        wav = decode_audio(pcm24, SR, log=lambda *_a: None)
        assert wav.shape == stereo.shape

    def test_downmixes_mono_to_stereo(self, tmp_path: Path) -> None:
        _ffmpeg_or_skip()
        mono = write_wav(tmp_path / "mono.wav", np.zeros(1000, np.float32), SR)
        wav = decode_audio(mono, SR)
        assert wav.shape == (1000, 2)

    def test_resamples_to_the_requested_rate(self, tmp_path: Path, stereo: np.ndarray) -> None:
        _ffmpeg_or_skip()
        src = write_wav(tmp_path / "r.wav", stereo, SR)
        wav = decode_audio(src, 22050)
        assert abs(wav.shape[0] - stereo.shape[0] // 2) <= 2

    def test_corrupt_input_surfaces_ffmpeg_stderr(self, tmp_path: Path) -> None:
        _ffmpeg_or_skip()
        broken = tmp_path / "broken.wav"
        broken.write_bytes(b"RIFF\x00\x00\x00\x00WAVE not really a wav")
        with pytest.raises(AudioIOError) as excinfo:
            decode_audio(broken, SR)
        text = str(excinfo.value)
        # ★ 报错必须带上 ffmpeg 自己的 stderr 与完整命令，否则 UI 上只有
        #   「返回非零」四个字，用户无从下手
        assert "ffmpeg 解码失败" in text
        assert "命令：" in text
        assert "ffmpeg 输出：" in text

    def test_odd_byte_count_is_reported_as_corruption(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """余数不为 0 说明流被截断，必须报错而不是 reshape 出错后崩溃。"""

        class _Proc:
            returncode = 0
            stdout = b"\x00" * 12  # 12 / 8 不整除
            stderr = b""

        monkeypatch.setattr(audio_io.subprocess, "run", lambda *_a, **_k: _Proc())
        with pytest.raises(AudioIOError, match="数据已损坏"):
            decode_audio(_touch(tmp_path / "x.wav"), SR, ffmpeg=_touch(tmp_path / "ffmpeg"))


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# rms / mix_error_pct
# ══════════════════════════════════════════════════════════════════════════════
class TestRms:
    def test_matches_the_closed_form(self) -> None:
        x = np.array([[3.0, 4.0], [0.0, 0.0]])
        assert rms(x) == pytest.approx(2.5)  # sqrt((9+16)/4)

    def test_empty_is_zero_not_nan(self) -> None:
        assert rms(np.zeros((0, 2))) == 0.0

    def test_silence_is_zero(self) -> None:
        assert rms(np.zeros((100, 2))) == 0.0

    def test_integers_are_accepted(self) -> None:
        assert rms(np.array([[3, 4]], dtype=np.int16)) == pytest.approx(np.sqrt(12.5))


class TestMixErrorPct:
    def test_identical_signals_give_zero(self, stereo: np.ndarray) -> None:
        assert mix_error_pct(stereo, stereo) == pytest.approx(0.0)

    def test_difference_is_relative_to_the_reference_rms(self) -> None:
        ref = np.full((100, 2), 0.5)
        err = np.full((100, 2), 0.55)
        assert mix_error_pct(err, ref) == pytest.approx(10.0)

    def test_silent_reference_returns_zero_not_nan(self) -> None:
        """★ 分母为 0 时返回 0，UI 才不会打出 NaN%。"""
        silent = np.zeros((10, 2))
        out = mix_error_pct(np.ones((10, 2)), silent)
        assert out == 0.0
        assert np.isfinite(out)


# ══════════════════════════════════════════════════════════════════════════════
# 导出面
# ══════════════════════════════════════════════════════════════════════════════
def test_exports_are_stable() -> None:
    assert audio_io.__all__ == [
        "AudioIOError",
        "decode_audio",
        "write_wav",
        "rms",
        "mix_error_pct",
    ]
    for name in audio_io.__all__:
        assert hasattr(audio_io, name), f"__all__ 里的 {name} 不存在"


def test_decode_does_not_use_check_output_or_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """★ ffmpeg 调用必须用参数列表 + capture_output，绝不能拼 shell 字符串。

    输入文件名带日文与全角字符，字符串拼接命令在 Windows 上正是注入与
    引号断裂的重灾区。
    """
    seen: dict[str, object] = {}

    class _Proc:
        returncode = 0
        stdout = b"\x00" * 16
        stderr = b""

    def spy(cmd, **kwargs):  # type: ignore[no-untyped-def]
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(audio_io.subprocess, "run", spy)
    decode_audio(_touch(tmp_path / UNICODE_NAME), SR, ffmpeg=_touch(tmp_path / "ffmpeg"))

    cmd = seen["cmd"]
    assert isinstance(cmd, list)
    assert isinstance(cmd[0], str) and cmd[0].endswith("ffmpeg")
    assert "shell" not in seen["kwargs"]
    assert seen["kwargs"].get("capture_output") is True
    assert subprocess.list2cmdline(cmd) == subprocess.list2cmdline(list(cmd))


def test_decode_commands_pin_the_model_format(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """解码参数必须钉死成 f32le / pcm_f32le / -ac 2 / -ar sr。"""
    seen: dict[str, object] = {}

    class _Proc:
        returncode = 0
        stdout = b"\x00" * 16
        stderr = b""

    def spy(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        seen["cmd"] = list(cmd)
        return _Proc()

    monkeypatch.setattr(audio_io.subprocess, "run", spy)
    decode_audio(_touch(tmp_path / "a.wav"), SR, ffmpeg=_touch(tmp_path / "ffmpeg"))

    cmd: list[str] = seen["cmd"]  # type: ignore[assignment]
    assert cmd[cmd.index("-f") + 1] == "f32le"
    assert cmd[cmd.index("-acodec") + 1] == "pcm_f32le"
    assert cmd[cmd.index("-ac") + 1] == "2"
    assert cmd[cmd.index("-ar") + 1] == str(SR)
    # 末尾必须是 "-"（写到 stdout），否则读到的是文件路径而不是音频流
    assert cmd[-1] == "-"
    assert "-nostdin" in cmd
