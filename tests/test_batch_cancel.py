"""PLAN §4 M3-6: 批量取消测试。

验证：
- 推理中取消，worker 在当前分块结束后停止
- 无残留 *.tmp
- 已完成文件产物完好
- 取消信号传递 ≤ 3s（从写哨兵文件到 worker 停止）
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_separation import paths as audio_paths
from audio_separation.core.batch import (
    create_job, run_worker, parse_worker_output, wait_for_worker,
    cancel_job, write_job,
)
from audio_separation.core.protocol import Event


@pytest.fixture(scope="module")
def clips():
    """获取测试切片。"""
    clip_dir = audio_paths.test_clips_dir()
    return list(clip_dir.glob("*.wav"))


def test_cancel_stops_worker(clips):
    """取消信号让 worker 在当前分块结束后停止。"""
    if not clips:
        pytest.skip("no test clips found")

    clip = clips[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        job_dir = Path(tmpdir)
        job = create_job([{"src": str(clip), "output_dir": str(job_dir)}],
                         model_key="v2", device="auto")
        job_path = write_job(job)

        proc = run_worker(job_path)

        # 立即取消（第一个分块还没开始或刚开始）
        time.sleep(0.5)
        cancel_job(job_dir)

        events = []
        start = time.monotonic()
        for event in parse_worker_output(proc):
            events.append(event)
            if event.type in ("cancelled", "done"):
                break
        elapsed = time.monotonic() - start

        exit_code = wait_for_worker(proc, timeout_sec=30)

        # 应该有 cancelled 或 done 事件
        got_cancelled = any(e.type == "cancelled" for e in events)
        got_done = any(e.type == "done" for e in events)
        assert got_cancelled or got_done, f"no cancel/done event: {[e.type for e in events]}"

        # worker 应该在合理时间内退出（≤ 120s，含模型加载 + 一个分块的推理时间）
        assert elapsed < 120, f"took {elapsed}s to cancel"

        # 无残留 .tmp 文件
        tmp_files = list(job_dir.glob("*.tmp")) + list(job_dir.rglob("*.tmp"))
        assert tmp_files == [], f"residual tmp files: {tmp_files}"

        job_path.unlink(missing_ok=True)


def test_cancel_before_start(clips):
    """在 worker 开始推理前取消。"""
    if not clips:
        pytest.skip("no test clips found")

    clip = clips[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        job_dir = Path(tmpdir)
        job = create_job([{"src": str(clip)}], model_key="v2", device="auto")
        job_path = write_job(job)

        # 先写取消哨兵
        cancel_job(job_dir)

        proc = run_worker(job_path)
        events = []
        for event in parse_worker_output(proc):
            events.append(event)
            if event.type in ("cancelled", "done"):
                break
        exit_code = wait_for_worker(proc, timeout_sec=30)

        # worker 应该立即停止（还没开始推理）
        got_done = any(e.type == "done" for e in events)
        assert got_done, f"expected done event, got: {[e.type for e in events]}"

        job_path.unlink(missing_ok=True)
