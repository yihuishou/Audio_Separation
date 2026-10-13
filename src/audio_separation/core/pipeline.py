# -*- coding: utf-8 -*-
"""单文件分离全流程：解码 → 推理 → 落盘 → mix 校验。

## 产物布局

```
<output_dir>/<输入文件 stem>/
    vocals.wav          模型输出的人声
    instrumental.wav    原轨 − vocals（残差伴奏，见下）
    mix_check.wav       vocals + instrumental，用于校验 == 原轨
    result.json         耗时、RMS、mix 误差百分比
```

### ★ 为什么 instrumental 是减法而不是第二个 stem

模型的 `num_stems: 1`，`training.target_instrument: vocals`
（`models/bs_roformer_revive/*.yaml`）。它只输出人声轨，
伴奏轨没有经过任何模型处理，是**原轨减人声**得到的残差。
所以 instrumental 里会带一点人声泄漏的补集 —— 这是单 stem 模型的
固有性质，不是 bug。UI 上必须写明这一点，否则用户会以为伴奏被"识别"了。

### ★ 为什么还要 mix_check.wav

`mix_check = vocals + instrumental`，理论上恒等于原轨。但四舍五入、
窗口边缘、int16 量化都可能让它偏离。写出来让用户能自己 A/B 验证，
同时 `mix_error_pct()` 给出数值判据。相对误差 < 1e-3 才算通过。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .. import paths
from .audio_io import decode_audio, mix_error_pct, rms, write_wav
from .config import TRAINED_SAMPLE_RATE, load_config_by_key
from .separator import BlockProgress, Cancelled, chunk_starts, separate_vocals

LogFn = Callable[..., None]
ProgressFn = Callable[[BlockProgress], None]
CancelFn = Callable[[], bool]

#: mix 相对 RMS 误差上限（%）。超过说明加窗归一化或残差构造出了问题。
MIX_ERROR_LIMIT_PCT = 1e-3


class PipelineError(RuntimeError):
    """全流程失败。消息里必带可直接操作的排查信息。"""


@dataclass
class SeparateResult:
    """一次单文件分离的完整结果。"""

    src: Path
    out_dir: Path
    vocals_path: Path
    instrumental_path: Path
    mix_check_path: Path
    result_json: Path
    elapsed_sec: float
    mix_error_pct: float
    blocks: int
    cache_hit: bool = False
    rms_vocals: float = 0.0
    rms_instrumental: float = 0.0
    rms_src: float = 0.0
    #: 本次实际用的模型（可能由调用方传入）。★ 不进 `as_dict()`：
    #: 610 MB 的对象不能被 `result.json` 序列化，但批量处理要靠它避免重载权重。
    model: object | None = field(default=None, repr=False, compare=False)
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """mix 校验是否通过。"""
        return self.mix_error_pct < MIX_ERROR_LIMIT_PCT

    def as_dict(self) -> dict:
        return {
            "src": str(self.src),
            "out_dir": str(self.out_dir),
            "vocals": str(self.vocals_path),
            "instrumental": str(self.instrumental_path),
            "mix_check": str(self.mix_check_path),
            "elapsed_sec": round(self.elapsed_sec, 3),
            "mix_error_pct": self.mix_error_pct,
            "mix_ok": self.ok,
            "blocks": self.blocks,
            "cache_hit": self.cache_hit,
            "rms_src": self.rms_src,
            "rms_vocals": self.rms_vocals,
            "rms_instrumental": self.rms_instrumental,
            **self.extra,
        }


def output_dir_for(src: Path, output_root: Path | str | None = None) -> Path:
    """输入文件对应的产物目录。

    ★ 目录名由输入 stem 派生 ⇒ 必然带日文与全角『』。
      Windows 建目录/写盘/删除都实测通过（PLAN R12 已降为低风险），
      所以不做「拷到纯 ASCII 临时名」绕行。
    """
    root = Path(output_root) if output_root is not None else paths.output_dir()
    return root / src.stem


def separate_file(
    src: Path | str,
    *,
    model_key: str = "v2",
    output_root: Path | str | None = None,
    device: str = "auto",
    dtype: str = "float32",
    model: object | None = None,
    use_cache: bool = True,
    block_timeout_sec: float = 0.0,
    write_mix_check: bool = True,
    stems: tuple[str, ...] = ("vocals", "instrumental"),
    log: LogFn = lambda *_a: None,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> SeparateResult:
    """分离一个音频文件。

    Args:
        src: 输入音频，任意 ffmpeg 能解的格式。
        model_key: 模型键（`paths.list_models()` 里的 `v2` / `v3`）。
        output_root: 产物根目录，默认 `AUDIO_SEPARATION_OUTPUT`（即 `output/separated/`）。
        device: `auto` / `cuda`。★**`auto` 是唯一该用的值**（§2.14）：
            它跑一次 `probe_device()`，GPU 不可用就抛 `PipelineError` 终止。
            `cpu` 不被接受。
        dtype: `float32` / `float16`（§2.5）。
        model: **已构建好的模型**。批量处理时由调用方构建一次复用，
            避免每文件重载 610 MB 权重。None 时本函数自行构建。
        use_cache: 命中 `<out>/vocals.npz` 则跳过推理。
        block_timeout_sec: 分块看门狗阈值；0 = 关闭（§2.9）。
        write_mix_check: 是否写 `mix_check.wav`。
        stems: 落盘哪些轨，取值 `vocals` / `instrumental`。
            ★ 两轨都必然会被算出来（伴奏是残差），这个参数只控制写不写盘，
              省的是磁盘与时间，不是计算量。
        log: 日志回调。
        on_progress: 分块进度回调。
        should_cancel: 返回 True 即在块间取消，抛 `Cancelled`。

    Raises:
        PipelineError: 输入/输出/配置错误。
        Cancelled: 用户取消（已落盘的文件不受影响）。
    """
    src = Path(src)
    if not src.exists():
        raise PipelineError(f"输入不存在：{src}")
    if src.stat().st_size == 0:
        raise PipelineError(f"输入是空文件：{src}")

    out_dir = output_dir_for(src, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "vocals.npz"
    vocals_path = out_dir / "vocals.wav"
    instrumental_path = out_dir / "instrumental.wav"
    mix_check_path = out_dir / "mix_check.wav"
    result_json = out_dir / "result.json"

    t0 = time.monotonic()
    log(f"[{src.name}] 输出目录：{out_dir}")

    wav = decode_audio(src, TRAINED_SAMPLE_RATE, log=log)
    log(f"  · 原轨 {wav.shape[0]} 样本，RMS {rms(wav):.6f}")

    vocals: np.ndarray | None = None
    cache_hit = False

    # ★块数必须在**缓存查询之前**算好，位置很关键。
    #   `blocks` 平时由 `_record` 从进度回调实录，但缓存命中路径根本不调
    #   `separate_vocals`，回调一次都不触发 → blocks 停在 0，而 result.json 与 UI
    #   都要靠它显示「共 15 块」，否则缓存命中会被显示成「0 块」。
    #   放在 `if vocals is None:` 里没用 —— 缓存命中时不进那个分支。
    #   `load_config_by_key()` 只解析 YAML（不加载 610 MB 权重），提前调用开销可忽略。
    #   推理路径上 `_record` 会用实录值覆盖它，两者必然一致：`chunk_starts`
    #   就是 `separate_vocals` 内部用的那个函数。
    cfg = load_config_by_key(model_key)
    blocks = len(
        chunk_starts(
            wav.shape[0],
            cfg["audio"]["chunk_size"],
            cfg["inference"]["num_overlap"],
        )
    )

    def _record(p: BlockProgress) -> None:
        nonlocal blocks
        blocks = p.total
        if on_progress is not None:
            on_progress(p)

    if use_cache and cache_path.exists():
        try:
            with np.load(cache_path) as data:
                cached = data["vocals"]
            if cached.shape == wav.shape:
                vocals = cached.astype(np.float32)
                cache_hit = True
                log(f"  · 命中缓存 {cache_path.name}，跳过推理")
            else:
                log(f"  · 缓存形状 {cached.shape} 与原轨 {wav.shape} 不符，忽略缓存")
        except (OSError, KeyError, ValueError) as exc:
            log(f"  · 缓存损坏（{exc}），重新推理")

    resolved_device: str | None = None
    if vocals is None:
        # ★解析必须在这里、且只做一次，放在 `if model is None` 之外。
        #   两个理由：
        #   1) 批量复用时调用方自带 model，上面那个分支不进 —— 若把解析挪进去，
        #      走复用路径就完全跳过了 GPU 可用性检查，§2.14 的终止语义形同虚设；
        #   2) `getattr(model, "device", device)` 是个陷阱：nn.Module 没有 device 属性，
        #      于是静默回退成调用方传的原始值（"auto"），一路带到 .to() 才炸出一个
        #      谁也看不懂的 "expected one of cpu, cuda, ... device string: auto"。
        resolved_device = _resolve_device(device, log=log)

        if model is None:
            from .model import build_model  # 延迟：让 --dry-run 与纯解码路径不必加载权重

            log(f"  · 构建模型 {model_key}（权重 {cfg['_weights']}）")
            model = build_model(cfg, device=resolved_device, log=log)

        log(f"  · 分离中（device={resolved_device}，dtype={dtype}）")
        vocals = separate_vocals(
            model,  # type: ignore[arg-type]
            wav,
            TRAINED_SAMPLE_RATE,
            chunk=cfg["audio"]["chunk_size"],
            num_overlap=cfg["inference"]["num_overlap"],
            device=resolved_device,
            dtype=dtype,
            on_progress=_record,
            should_cancel=should_cancel,
            block_timeout_sec=block_timeout_sec,
            log=log,
        )
        if use_cache:
            np.savez(cache_path, vocals=vocals)

    # ★ 伴奏是残差，不是第二个 stem（num_stems=1，target_instrument=vocals）
    instrumental = wav.astype(np.float32) - vocals
    mix_check = vocals + instrumental

    if "vocals" in stems:
        write_wav(vocals_path, vocals, TRAINED_SAMPLE_RATE)
    if "instrumental" in stems:
        write_wav(instrumental_path, instrumental, TRAINED_SAMPLE_RATE)
    if write_mix_check:
        write_wav(mix_check_path, mix_check, TRAINED_SAMPLE_RATE)

    err = mix_error_pct(mix_check, wav)
    elapsed = time.monotonic() - t0
    result = SeparateResult(
        src=src,
        out_dir=out_dir,
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        mix_check_path=mix_check_path,
        result_json=result_json,
        elapsed_sec=elapsed,
        mix_error_pct=err,
        blocks=blocks,
        cache_hit=cache_hit,
        rms_src=rms(wav),
        rms_vocals=rms(vocals),
        rms_instrumental=rms(instrumental),
        model=model,
        # ★缓存命中时 resolved_device 是 None，写进 result.json 就是 null ——
        #   这是诚实值：本次运行确实没有做任何 GPU 计算，不该记一个从缓存里
        #   抄来的设备名。（代价是缓存命中那次不跑探测，省下约 5 s。）
        extra={"model_key": model_key, "device": resolved_device, "dtype": dtype},
    )

    tmp = result_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, result_json)

    verdict = "PASS" if result.ok else "FAIL"
    log(
        f"  · mix 校验 {verdict}：相对 RMS 误差 {err:.3e}%"
        f"（上限 {MIX_ERROR_LIMIT_PCT}%）· 用时 {elapsed:.1f}s"
    )
    if not result.ok:
        log(f"  · ★ 误差超限，检查 {mix_check_path.name} 是否能听出接缝")
    return result


def _resolve_device(device: str, *, log: LogFn) -> str:
    """解析推理设备。★PLAN §2.14：GPU-only，探测不通过就抛错终止。

    `probe_device()` 的 `backend` 取值只有 rocm / cuda，两者**都**映射到 torch 的
    `"cuda"` 设备串 —— ROCm wheel 不引入 `rocm` 这个设备名，装完 `torch.cuda.*`
    一切照旧。可用的显示名在 `device_name`，不是 `device`。

    不要在这里加 CPU 兜底：`build_model()` 的 610 MB 权重 + 11 s 音频块在 CPU 上
    比 GPU 慢约 50 倍，"能跑完"等于"不可用"，而且用户不会知道自己拿到的结果是
    在哪个后端上算的。详见 PLAN §2.14。
    """
    from .env_check import probe_device

    if device != "auto":
        if device != "cuda":
            # 不是内部错误就是调用方没跟上 §2.14 的口径，直接挡掉而不是照做
            raise PipelineError(
                f"不支持的设备「{device}」。本工具只支持 GPU 推理"
                f"（PLAN §2.14），请传 auto 或 cuda。"
            )
        return "cuda"

    probe = probe_device()
    if not probe.get("ok"):
        raise PipelineError(
            f"GPU 不可用，已终止：{probe.get('error') or '未知原因'}\n"
            f"本工具不提供 CPU 模式（PLAN §2.14）。"
        )

    backend = str(probe.get("backend", "")).lower()
    name = probe.get("device_name") or "未知设备"
    if backend in ("rocm", "cuda"):
        log(f"  · 设备：{name}（后端 {backend}，torch 设备串 cuda）")
        return "cuda"

    # probe 报 ok=True 却给出别的后端 —— 与其猜，不如当成探测不可信
    raise PipelineError(
        f"GPU 探测结果不可信：backend={backend!r}。已终止（PLAN §2.14）。"
    )


__all__ = [
    "MIX_ERROR_LIMIT_PCT",
    "PipelineError",
    "SeparateResult",
    "output_dir_for",
    "separate_file",
]
