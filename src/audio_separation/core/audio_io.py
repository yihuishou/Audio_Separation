# -*- coding: utf-8 -*-
"""音频读写（§2.4）。

## 为什么解码走 ffmpeg 而不是 soundfile

输入可能是 mp3/flac/m4a 等任意格式，`soundfile`（libsndfile）**解不了 mp3**，
而 librosa 虽能解但会把整轨以 float64 读进内存（一条 930 s 的测试素材就是
800 MB 级别），并且它内部的 audioread/auralias 在部分 Windows 环境上
行为不稳。ffmpeg 是项目内嵌的固定二进制，行为确定、内存占用是流式的。

## 为什么写盘用 scipy.io.wavfile 而不是 soundfile

参考脚本注释原文：**避开 soundfile libsndfile 1.2.2 bug**。
`scipy.io.wavfile` 写 16bit PCM 是纯 numpy 路径，没有 C 扩展版本耦合。

## ★ Unicode 路径已实测可用

`test_audio/` 里的文件名是 `1.第1章『耳なめ手コキオナサポ』.wav`，
产物名由输入 stem 派生、必然带日文与全角『』。
已实测（2026-10-13，见 PLAN §1.2 第 3 点 / R12）：ffmpeg 在 Windows 上
**按宽字符正确解析 argv**，输入输出全带日文+全角字符仍 `exit=0`。
所以这里**不实现**「拷到纯 ASCII 临时名」或 `-i pipe:0` 的绕行方案。
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from .. import paths

#: 16bit PCM 的满量程系数
_INT16_MAX = 32767.0

LogFn = Callable[..., None]


def _noop(*_args: object, **_kwargs: object) -> None:
    """默认日志函数（吞掉一切）。"""


class AudioIOError(RuntimeError):
    """音频读写失败。消息里必带可直接粘进终端排查的信息。"""


# ══════════════════════════════════════════════════════════════════════════════
# 解码
# ══════════════════════════════════════════════════════════════════════════════
def decode_audio(
    src: Path | str,
    sample_rate: int,
    *,
    ffmpeg: Path | str | None = None,
    log: LogFn = _noop,
) -> np.ndarray:
    """把任意格式音频解码成 `(n, 2)` 的 float32 立体声数组。

    Args:
        src: 输入文件路径。**允许含日文/全角字符**，ffmpeg 按宽字符处理。
        sample_rate: 目标采样率（模型固定 44100）。
        ffmpeg: ffmpeg 可执行文件；None 时走 `paths.find_ff()`。
        log: 日志回调，签名同 `print`。

    Returns:
        `(n, 2)` float32，取值区间 [-1, 1]。`n == 0` 表示输入为空。

    Raises:
        AudioIOError: ffmpeg 缺失、输入不存在、或解码失败（含 ffmpeg 的 stderr）。
    """
    src = Path(src)
    exe = _resolve_ffmpeg(ffmpeg)

    if not src.exists():
        raise AudioIOError(f"输入不存在：{src}")
    if src.stat().st_size == 0:
        raise AudioIOError(f"输入是空文件：{src}")

    cmd = [
        str(exe), "-v", "error", "-nostdin", "-i", str(src),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sample_rate), "-",
    ]
    log(f"  · ffmpeg 解码 → {sample_rate} Hz 立体声 f32le")
    # ★ 不能用 check=True：那样抛出的 CalledProcessError 不带 stderr 的可读内容，
    #   UI 面板只会显示「命令返回非零」，用户无从下手。这里自己捕获并附上原始报错。
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioIOError(
            f"ffmpeg 解码失败（退出码 {proc.returncode}）：{src}\n"
            f"命令：{' '.join(cmd)}\n"
            f"ffmpeg 输出：\n{detail or '（无 stderr）'}"
        )

    raw = proc.stdout
    # f32le 固定 4 字节/样本/声道；余数不为 0 说明 ffmpeg 输出被截断
    if len(raw) % (4 * 2) != 0:
        raise AudioIOError(
            f"ffmpeg 输出的字节数 {len(raw)} 不是 4×2 的整数倍，数据已损坏。"
        )
    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    log(f"  · 解码得到 {wav.shape[0]} 样本 = {wav.shape[0] / sample_rate:.1f}s")
    return wav


# ══════════════════════════════════════════════════════════════════════════════
# 编码
# ══════════════════════════════════════════════════════════════════════════════
def write_wav(path: Path | str, wav: np.ndarray, sample_rate: int) -> Path:
    """把 `(n, 2)` float32 数组写成 16bit PCM WAV。

    ★ 先写 `.tmp` 再 `os.replace()` 原子替换 —— 中途崩掉 / 被取消时，
      目标位置要么是旧的完整文件，要么不存在，**绝不会留下半截 WAV**
      （下游按目录扫描产物的逻辑才不会把半截文件当成品）。

    Args:
        path: 目标路径，父目录必须已存在（★本函数不 mkdir，见 PLAN §2.10）。
        wav: `(n, 2)` 或 `(n,)` float32。
        sample_rate: 采样率。

    Returns:
        写入后的路径。

    Raises:
        AudioIOError: 形状不对或写入失败。
    """
    import scipy.io.wavfile as sfw

    path = Path(path)
    arr = np.asarray(wav)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise AudioIOError(f"wav 必须是 (n,) 或 (n, channels)，实际 ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise AudioIOError("wav 是空数组，写不出有效 WAV")

    pcm = np.clip(arr, -1.0, 1.0)
    # ★ 用 32767 而不是 32768：-1.0 → -32767，映射严格落在 int16 区间内，
    #   用 32768 会让 -1.0 溢出成 +32767（正负翻转）。
    pcm = (pcm * _INT16_MAX).astype("<i2")

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        sfw.write(str(tmp), int(sample_rate), pcm)
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)  # 别把半截文件留在磁盘上
        raise AudioIOError(f"写入失败：{path}\n{exc}") from exc
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════════════════════════════════════
def _ensure_executable(exe: Path) -> Path:
    """确保 ffmpeg 可执行。

    ★ PyInstaller 的 `--add-binary` **不保留文件权限位**（官方 issue #3926），
    打进 bundle 的 Linux ffmpeg 会丢掉 +x，表现为 "Permission denied"。
    这里补一次 chmod —— Windows 无所谓权限位，直接跳过。
    """
    if platform.system() == "Windows":
        return exe
    try:
        if not os.access(exe, os.X_OK):
            exe.chmod(exe.stat().st_mode | 0o755)
    except OSError:
        # 只读文件系统 / 无权限：留给调用方报更具体的错
        pass
    return exe


def _resolve_ffmpeg(ffmpeg: Path | str | None) -> Path:
    """定位 ffmpeg。显式传入优先，其次 `paths.find_ff()`（内置目录 → PATH → rglob）。"""
    if ffmpeg is not None:
        exe = Path(ffmpeg)
        if not exe.exists():
            raise AudioIOError(
                f"指定的 ffmpeg 不存在：{exe}\n"
                f"★ 应放在 {paths.ffmpeg_bin()}（内嵌二进制，不依赖系统 PATH）"
            )
        return _ensure_executable(exe)
    exe = paths.find_ff()
    if exe is None:
        raise AudioIOError(
            f"找不到 ffmpeg。\n"
            f"★ 期望位置：{paths.ffmpeg_bin()}\n"
            f"★ 该文件是项目内嵌的必需资源（见 PLAN §2.13 F 组），"
            f"不是可选依赖，也不是检测项（R20）。"
        )
    return _ensure_executable(exe)


def rms(x: np.ndarray) -> float:
    """整体 RMS。分离结果的音量指标与 mix 校验都用它。"""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a ** 2)))


def mix_error_pct(mix: np.ndarray, reference: np.ndarray) -> float:
    """`mix` 与 `reference` 的 RMS 相对误差（%）。

    ★ 分母加 1e-9 兜底：参考轨全静音时相对误差无意义，返回 0 而不是 inf/NaN。
    """
    ref = rms(reference)
    if ref <= 0.0:
        return 0.0
    return rms(np.asarray(mix, dtype=np.float64) - np.asarray(reference, dtype=np.float64)) / ref * 100.0


__all__ = [
    "AudioIOError",
    "decode_audio",
    "write_wav",
    "rms",
    "mix_error_pct",
]
