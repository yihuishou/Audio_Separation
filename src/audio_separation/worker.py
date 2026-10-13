# -*- coding: utf-8 -*-
"""推理 worker 进程（PLAN §2.3 / §4 M3-2）。

用法：
    python -m audio_separation.worker --job job.json

职责：
- 读 job.json，按 action 执行 start/cancel
- 逐文件调用 core.pipeline.separate_file
- 每步通过 JSON Lines 事件流（stdout）通知 GUI
- 分块间检查取消哨兵文件
- 看门狗超时由 pipeline 内部处理（os._exit(3)）

★ 本进程 import torch/bs_roformer —— 它是真正的推理进程。
  GUI 进程（audio_separation.ui.app）不 import，只通过 JSON Lines 与之通信。
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from audio_separation import paths
from audio_separation.core.config import load_config_by_key
from audio_separation.core.model import build_model
from audio_separation.core.pipeline import separate_file, _resolve_device
from audio_separation.core.protocol import Event, Job, validate_event

CANCEL_SENTINEL = ".cancel"


def _make_log(emit: Callable[[Event], None]) -> Callable[..., None]:
    """把 pipeline 的 log 回调转成 JSON Lines 事件。"""
    def log(*args):
        msg = " ".join(str(a) for a in args)
        emit(Event.log(msg))
    return log


def _check_cancel(job_dir: Path) -> bool:
    """检查取消哨兵文件。"""
    sentinel = job_dir / CANCEL_SENTINEL
    return sentinel.exists()


def _emit(event: Event) -> None:
    """发送一条 JSON Lines 事件到 stdout。"""
    missing = validate_event(event)
    if missing:
        print(f"worker: invalid event {event.type}: missing {missing}",
              file=sys.stderr, flush=True)
        return
    print(event.to_line(), flush=True)


def run(job_path: str) -> int:
    """执行 job.json 指定的任务，返回退出码。"""
    job_file = Path(job_path)
    if not job_file.exists():
        print(f"worker: job file not found: {job_path}", file=sys.stderr, flush=True)
        return 1

    job_text = job_file.read_text(encoding="utf-8")
    job = Job.from_json(job_text)
    job_dir = job_file.parent

    if job.action == "cancel":
        # cancel action 只是退出，实际取消由 GUI 写哨兵文件触发
        return 0

    if not job.inputs:
        _emit(Event.log("no input files"))
        _emit(Event.done(0, 0, 0, 0))
        return 0

    # 清理上一次的取消哨兵
    sentinel = job_dir / CANCEL_SENTINEL
    if sentinel.exists():
        sentinel.unlink()

    total = len(job.inputs)
    ok = 0
    failed = 0
    cancelled = 0

    # 解析设备（"auto" → "cuda"，或终止）
    _emit(Event.log(f"resolving device: {job.device}"))
    resolved_device = _resolve_device(job.device, log=lambda *a: _emit(Event.log("  " + " ".join(map(str, a)))))
    _emit(Event.log(f"resolved device: {resolved_device}"))

    # 模型只构建一次，批量文件复用（PLAN §2.3）
    model = None
    cfg = None
    try:
        cfg = load_config_by_key(job.model_key)
        _emit(Event.log(f"building model {job.model_key}"))
        model = build_model(cfg, device=resolved_device, log=lambda *a: _emit(Event.log("  " + " ".join(map(str, a)))))
        _emit(Event.log(f"model built ({sum(p.numel() for p in model.parameters()):,} params)"))
    except Exception as exc:
        _emit(Event.fatal(f"model build failed: {exc}"))
        return 1

    try:
        for i, inp in enumerate(job.inputs):
            src = Path(inp.src)
            if not src.exists():
                _emit(Event.file_done(str(src), False, 0.0, None, ""))
                failed += 1
                continue

            # 检查取消（在开始下一个文件前）
            if _check_cancel(job_dir):
                _emit(Event.log(f"cancelling at file {i + 1}/{total}"))
                cancelled = total - i
                break

            out_dir = Path(inp.output_dir) if inp.output_dir else None
            try:
                result = separate_file(
                    src=src,
                    model_key=job.model_key,
                    output_root=out_dir,
                    device=resolved_device,
                    dtype=job.dtype,
                    model=model,
                    block_timeout_sec=job.block_timeout_sec,
                    log=lambda *a: _emit(Event.log("  " + " ".join(map(str, a)))),
                    on_progress=lambda prog: _emit(Event.progress(
                        str(src.name), prog.done, prog.total, prog.eta_sec)),
                    should_cancel=lambda: _check_cancel(job_dir),
                )
                if result.ok:
                    ok += 1
                else:
                    failed += 1
                _emit(Event.file_done(
                    str(src.name), result.ok, result.elapsed_sec,
                    result.mix_error_pct, str(result.out_dir)))
            except Exception as exc:
                failed += 1
                _emit(Event.file_done(str(src.name), False, 0.0, None, ""))
                _emit(Event.log(f"  error: {exc}"))

    except KeyboardInterrupt:
        _emit(Event.log("interrupted"))
        cancelled = total - ok - failed

    _emit(Event.done(total, ok, failed, cancelled))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audio separation worker process (JSON Lines protocol)")
    parser.add_argument("--job", required=True, help="path to job.json")
    args = parser.parse_args(argv)
    return run(args.job)


if __name__ == "__main__":
    sys.exit(main())
