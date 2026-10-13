# -*- coding: utf-8 -*-
"""命令行入口：`python -m audio_separation.core.cli -i <音频...>`

这是 GUI 底下真正干活的同一条代码路径 —— `worker.py`（M3）读 job.json
之后调的是同一个 `pipeline.separate_file()`。CLI 的存在有两个作用：
一是无 GUI 环境（服务器、ssh）下能跑；二是出问题时有个不经过 Qt 的最小复现。

## ★ 与参考脚本的取舍

| 参考脚本 | 本 CLI | 理由 |
|---|---|---|
| `--stems vocals,instrumental` | 保留，但值域限定为 `vocals` / `instrumental` | 两轨都是**必然算出来的**（伴奏是残差），这个开关只是「落盘哪些文件」，不是模型选择 |
| `chunk` / `overlap` / `batch` 覆盖开关 | **不提供** | §2.11：这些是模型常量，交叉耦合，用户改了不报错、只会在一次前向之后表现为「分离结果很差」 |
| 无时长护栏 | `--max-minutes`（默认 60）+ `--full-length` 绕过 | 一条 15 分钟的歌在 CPU 上要跑数小时，先拦下来比让它跑一整夜有用 |
| `os._exit(3)` 超时 | 默认动作仍是 `os._exit(3)`（§2.9） | CLI 没有父进程兜底，不自己退就永远卡在挂死的算子里 |

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 全部成功 |
| 1 | 有文件失败（参数错误、输入损坏、模型加载失败） |
| 3 | 分块超时被看门狗强杀（`watchdog.TIMEOUT_EXIT_CODE`） |
| 130 | 用户 Ctrl+C |
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable
from pathlib import Path

from .. import paths
from .audio_io import AudioIOError
from .config import ConfigError, TRAINED_SAMPLE_RATE, describe_config, load_config_by_key
from .pipeline import PipelineError, SeparateResult, separate_file
from .separator import Cancelled

LogFn = Callable[..., None]

#: 时长护栏默认值（分钟）。超过则拒绝执行，除非 `--full-length`。
DEFAULT_MAX_MINUTES = 60.0

_STEMS = ("vocals", "instrumental")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m audio_separation.core.cli",
        description="BS-RoFormer revive 人声分离（伴奏为残差，非独立 stem）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  # 先看看有什么模型、参数是什么\n"
            "  python -m audio_separation.core.cli --list-models\n"
            "  # 只算计划不加载权重\n"
            "  python -m audio_separation.core.cli -i a.mp3 --dry-run\n"
            "  # 正式跑，自动挑设备\n"
            "  python -m audio_separation.core.cli -i a.mp3 -o output/separated\n"
        ),
    )
    p.add_argument(
        "-i", "--input", type=Path, nargs="+", metavar="音频",
        help="输入音频，可给多个；格式由 ffmpeg 解码能力决定",
    )
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help=f"产物根目录（默认 {paths.output_dir()}，每个输入各自建子目录）",
    )
    p.add_argument(
        "-m", "--model", default="v2", metavar="KEY",
        help="模型键，用 --list-models 查看（默认 v2）",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda"],
        help=(
            "auto = 运行时探测 GPU（默认）；cuda 同时指 ROCm 与 CUDA（torch 设备串统一）。"
            "★没有 cpu 取值：GPU 不可用会立即终止，不做 CPU 回退（PLAN §2.14）"
        ),
    )
    p.add_argument(
        "--dtype", default="float32", choices=["float32", "float16"],
        help="★ ROCm 上 bf16/fp16 的 STFT 会掉精度，默认 float32",
    )
    p.add_argument(
        "--stems", default=",".join(_STEMS),
        help=f"落盘哪些轨，逗号分隔，值域 {'/'.join(_STEMS)}（默认两个都写）",
    )
    p.add_argument(
        "--no-mix-check", action="store_true",
        help="不写 mix_check.wav（省一份 16bit WAV 的磁盘与时间）",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="忽略并覆盖 <输出>/vocals.npz 缓存",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="只打印计划与耗时预估，不解码、不加载权重",
    )
    p.add_argument(
        "--block-timeout", type=float, default=300.0, metavar="秒",
        help="分块无进展超过这个秒数就强杀（默认 300；0 = 关闭看门狗）",
    )
    p.add_argument(
        "--max-minutes", type=float, default=DEFAULT_MAX_MINUTES, metavar="分钟",
        help=f"单文件时长护栏（默认 {DEFAULT_MAX_MINUTES:g}），超过则拒绝执行",
    )
    p.add_argument(
        "--full-length", action="store_true",
        help="关闭时长护栏，人工长跑逃生开关",
    )
    p.add_argument(
        "--list-models", action="store_true",
        help="列出可用模型与其关键参数后退出",
    )
    return p


def _print_models(log: LogFn) -> None:
    infos = paths.list_models()
    if not infos:
        log(f"没有找到任何模型。请确认权重放在 {paths.bs_roformer_revive_dir()}")
        return
    for info in infos:
        log(f"── {info.key} ──")
        for label, value in describe_config(load_config_by_key(info.key)):
            log(f"   {label:<12} {value}")


def _probe_duration(src: Path) -> float | None:
    """用 ffprobe 读时长（秒）。读不到返回 None，不阻断流程。"""
    import subprocess

    probe = paths.ffprobe_bin()
    if probe is None or not probe.exists():
        return None
    proc = subprocess.run(
        [str(probe), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(src)],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        return None


def _check_length(src: Path, max_minutes: float, full_length: bool, log: LogFn) -> bool:
    if full_length or max_minutes <= 0:
        return True
    seconds = _probe_duration(src)
    if seconds is None:
        log(f"  · 读不到时长，跳过护栏检查")
        return True
    minutes = seconds / 60.0
    if minutes <= max_minutes:
        return True
    log(
        f"  · ★ 拒绝执行：{src.name} 长 {minutes:.1f} 分钟，超过护栏 {max_minutes:g} 分钟\n"
        f"    分块推理的耗时随长度线性增长，CPU 上可能跑到天亮。\n"
        f"    确实要长跑请加 --full-length（逃生开关）。"
    )
    return False


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。返回进程退出码（`__main__` 里 `sys.exit(main())`）。"""
    args = build_parser().parse_args(argv)
    log = _make_log()

    if args.list_models:
        _print_models(log)
        return 0
    if not args.input:
        build_parser().print_help()
        return 1

    try:
        cfg = load_config_by_key(args.model)
    except ConfigError as exc:
        log(f"配置错误：{exc}")
        log(f"用 --list-models 看看有哪些模型")
        return 1

    missing = [s.strip() for s in args.stems.split(",") if s.strip()]
    bad = [s for s in missing if s not in _STEMS]
    if bad:
        log(f"--stems 含未知取值 {bad}；可用值只有 {list(_STEMS)}")
        log("★ 伴奏轨不是独立 stem（num_stems=1，target_instrument=vocals），")
        log("  它由「原轨 − 人声」得到，两轨必然都会算出来，这个开关只控制落盘。")
        return 1
    want = set(missing) or set(_STEMS)

    sources = [p for p in args.input]
    for src in sources:
        if not src.exists():
            log(f"输入不存在：{src}")
            return 1
        if src.stat().st_size == 0:
            log(f"输入是空文件：{src}")
            return 1

    if args.dry_run:
        return _dry_run(cfg, args, sources, want, log)

    rejected = [s for s in sources if not _check_length(s, args.max_minutes, args.full_length, log)]
    if rejected:
        return 1

    # 模型只构建一次，多文件复用（610 MB 权重不该每个文件重载一遍）
    model = None
    failed: list[Path] = []
    try:
        for src in sources:
            log(f"=== {src.name} ===")
            result = separate_file(
                src,
                model_key=args.model,
                output_root=args.output_dir,
                device=args.device,
                dtype=args.dtype,
                model=model,
                use_cache=not args.no_cache,
                block_timeout_sec=args.block_timeout,
                write_mix_check=not args.no_mix_check,
                log=log,
                stems=tuple(sorted(want)),
            )
            model = result.model or model
            for line in _report(result, want):
                log(line)
    except Cancelled:
        log("已取消。已完成的文件产物保留。")
        return 130
    except KeyboardInterrupt:
        log("已中断。")
        return 130
    except (PipelineError, AudioIOError, ConfigError, RuntimeError) as exc:
        log(f"失败：{exc}")
        failed.append(sources[0])
    finally:
        if model is not None:
            model = None

    return 1 if failed else 0


def _dry_run(cfg: dict, args: argparse.Namespace, sources: list[Path], want: set[str], log: LogFn) -> int:
    """只打印计划。★ 不 import torch、不加载权重。"""
    from .separator import chunk_starts

    chunk = cfg["audio"]["chunk_size"]
    overlap = cfg["inference"]["num_overlap"]
    sr = cfg["audio"]["sample_rate"]
    log(f"模型      {args.model}（{Path(cfg['_weights']).name}）")
    log(f"设备      {args.device}    精度 {args.dtype}")
    log(f"采样率    {sr}")
    log(f"分块      chunk={chunk}（{chunk / sr:.1f}s）  num_overlap={overlap}"
        f" ⇒ hop={chunk // (overlap + 1)}")
    log(f"落盘      {'、'.join(sorted(want))}"
        f"{'' if args.no_mix_check else ' + mix_check'}")
    log(f"缓存      {'关闭' if args.no_cache else f'{paths.output_dir()}/<曲名>/vocals.npz'}")
    log(f"输出根    {args.output_dir or paths.output_dir()}")
    log(f"看门狗    {'关闭' if args.block_timeout <= 0 else f'{args.block_timeout:g}s 无进展即强杀'}")
    log("")

    for src in sources:
        seconds = _probe_duration(src)
        if seconds is None:
            log(f"· {src.name}：读不到时长，跳过块数估算")
            continue
        total = int(seconds * sr)
        blocks = len(chunk_starts(total, chunk, overlap))
        est = seconds / 60.0
        log(f"· {src.name}")
        log(f"    {est:.1f} 分钟 ⇒ {total} 样本 ⇒ {blocks} 个分块")
        log(f"    → {paths.output_dir() if args.output_dir is None else args.output_dir}/{src.stem}/")
    log("")
    log("★ 这是 dry-run：没有解码、没有加载权重。")
    log("  ★ 伴奏轨是「原轨 − 人声」的残差，不是模型独立输出。")
    return 0


def _report(result: SeparateResult, want: set[str]) -> list[str]:
    lines = [f"  → {result.out_dir}"]
    if "vocals" in want:
        lines.append(f"    vocals.wav         RMS {result.rms_vocals:.5f}")
    if "instrumental" in want:
        lines.append(f"    instrumental.wav   RMS {result.rms_instrumental:.5f}")
    lines.append(
        f"    mix 校验 {'PASS' if result.ok else 'FAIL'}"
        f"：相对 RMS 误差 {result.mix_error_pct:.3e}%　用时 {result.elapsed_sec:.1f}s"
    )
    if result.cache_hit:
        lines.append("    （命中缓存，未重新推理）")
    return lines


class _Log:
    """最小日志协议。"""


def _make_log() -> _Log:
    import functools

    @functools.wraps(print)
    def log(*args: object, **kwargs: object) -> None:
        print(*args, flush=True, **kwargs)

    return log  # type: ignore[return-value]


__all__ = ["DEFAULT_MAX_MINUTES", "build_parser", "main"]


if __name__ == "__main__":
    sys.exit(main())
