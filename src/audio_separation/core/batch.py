# -*- coding: utf-8 -*-
"""批量队列管理与 worker 输出解析（PLAN §4 M3-3）。

职责：
- 生成 job.json
- 从 worker 的 stdout 流式解析 JSON Lines 事件
- 提供取消接口（写哨兵文件）

★ 本模块不依赖 torch —— GUI 侧与调度器都可以 import。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

from .. import paths
from .protocol import Event, Job, JobInput

WORKER_SCRIPT = None  # 延迟解析


def _resolve_worker_script() -> Path:
    """找到 worker.py 的路径。"""
    global WORKER_SCRIPT
    if WORKER_SCRIPT is not None:
        return WORKER_SCRIPT
    WORKER_SCRIPT = Path(__file__).parent.parent / "worker.py"
    return WORKER_SCRIPT


def create_job(
    inputs: list[dict[str, str]],
    model_key: str = "v2",
    device: str = "auto",
    dtype: str = "float32",
    block_timeout_sec: int = 300,
) -> Job:
    """从输入列表创建 Job 对象。

    inputs: [{"src": "/path/to/file.wav", "output_dir": "/path/to/out"}, ...]
    """
    job_inputs = [JobInput(src=i["src"], output_dir=i.get("output_dir")) for i in inputs]
    return Job(
        model_key=model_key,
        device=device,
        dtype=dtype,
        block_timeout_sec=block_timeout_sec,
        inputs=job_inputs,
        action="start",
    )


def write_job(job: Job, path: Path | None = None) -> Path:
    """写 job.json，返回路径。"""
    if path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="audio_sep_job_", suffix=".json", delete=False)
        path = Path(tmp.name)
        tmp.close()
    path.write_text(job.to_json(), encoding="utf-8")
    return path


def cancel_job(job_dir: Path) -> None:
    """通过写哨兵文件通知 worker 取消。"""
    sentinel = job_dir / ".cancel"
    sentinel.write_text("cancel", encoding="utf-8")


def run_worker(job_path: Path) -> subprocess.Popen:
    """启动 worker 子进程，返回 Popen 对象。"""
    worker_script = _resolve_worker_script()
    proc = subprocess.Popen(
        [sys.executable, str(worker_script), "--job", str(job_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc


def parse_worker_output(proc: subprocess.Popen) -> Iterator[Event]:
    """流式解析 worker 的 stdout JSON Lines 事件。"""
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = Event.from_line(line)
            if event is not None:
                yield event
        except json.JSONDecodeError as exc:
            print(f"batch: failed to parse worker output: {exc}\n{line!r}",
                  file=sys.stderr, flush=True)


def wait_for_worker(proc: subprocess.Popen, timeout_sec: int | None = None) -> int:
    """等待 worker 结束，返回退出码。"""
    try:
        return proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


def run_batch(
    inputs: list[dict[str, str]],
    model_key: str = "v2",
    device: str = "auto",
    dtype: str = "float32",
    block_timeout_sec: int = 300,
    max_retries: int = 2,
    job_dir: Path | None = None,
) -> Iterator[Event]:
    """运行批量分离，流式返回事件。

    带重试：worker 进程异常退出时重试最多 max_retries 次，指数退避。

    Args:
        job_dir: 作业目录。job.json 与取消哨兵 `.cancel` 都放在这里。
            ★不传时 job.json 落在系统临时目录，调用方拿不到路径 ——
            GUI 就没法写取消哨兵了。GUI 侧必须自己建目录并传进来。
    """
    job = create_job(inputs, model_key, device, dtype, block_timeout_sec)
    if job_dir is not None:
        job_dir.mkdir(parents=True, exist_ok=True)
        job_path = write_job(job, job_dir / "job.json")
    else:
        job_path = write_job(job)

    try:
        for attempt in range(max_retries + 1):
            if attempt > 0:
                backoff = 2 ** (attempt - 1)
                yield Event.log(f"retry {attempt}/{max_retries} after {backoff}s")
                time.sleep(backoff)

            proc = run_worker(job_path)
            try:
                for event in parse_worker_output(proc):
                    yield event

                exit_code = wait_for_worker(proc)
                if exit_code != 0:
                    yield Event.fatal(f"worker exited with code {exit_code}")
                break  # 成功或达到最大重试次数
            except Exception as exc:
                if attempt < max_retries:
                    yield Event.log(f"worker failed: {exc}")
                else:
                    raise
    finally:
        job_path.unlink(missing_ok=True)
        # 清理哨兵文件
        job_dir = job_path.parent
        sentinel = job_dir / ".cancel"
        sentinel.unlink(missing_ok=True)
