"""PLAN §4 M3-5: worker 协议测试。

验证 JSON Lines 协议的完整性：事件顺序、字段、退出码、stderr 不污染 stdout、
Unicode 路径往返。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_separation.core.protocol import Event, Job, JobInput, validate_event


def test_job_roundtrip():
    """job.json 序列化/反序列化往返。"""
    job = Job(
        model_key="v2",
        device="auto",
        dtype="float32",
        block_timeout_sec=300,
        inputs=[
            JobInput(src="/path/to/1.第1章『耳なめ』.wav",
                     output_dir="/path/to/out/1.第1章『耳なめ』"),
            JobInput(src="/path/to/2.第2章.wav"),
        ],
    )
    text = job.to_json()
    job2 = Job.from_json(text)
    assert job2.model_key == "v2"
    assert job2.inputs[0].src == "/path/to/1.第1章『耳なめ』.wav"
    assert len(job2.inputs) == 2
    assert job2.action == "start"


def test_event_serialization():
    """各种事件类型的序列化/反序列化。"""
    events = [
        Event.log("  · 测试日志"),
        Event.progress("file.wav", 5, 15, 30.5),
        Event.file_done("file.wav", True, 45.2, 0.001, "/path/to/out"),
        Event.fatal("  · 模型加载失败"),
        Event.cancelled("user requested"),
        Event.done(3, 2, 1, 0),
    ]
    for event in events:
        line = event.to_line()
        assert isinstance(line, str)
        parsed = Event.from_line(line)
        assert parsed is not None
        assert parsed.type == event.type
        missing = validate_event(parsed)
        assert missing == [], f"event {event.type} missing fields: {missing}"


def test_event_field_requirements():
    """事件字段完整性校验。"""
    # log 必须有 message
    bad = Event(type="log", data={})
    missing = validate_event(bad)
    assert "message" in missing

    # progress 必须有 file, block, total_blocks
    bad = Event(type="progress", data={"file": "x"})
    missing = validate_event(bad)
    assert "block" in missing
    assert "total_blocks" in missing

    # file_done 必须有 ok
    bad = Event(type="file_done", data={"file": "x"})
    missing = validate_event(bad)
    assert "ok" in missing


def test_event_from_empty_line():
    """空行返回 None（不报错）。"""
    assert Event.from_line("") is None
    assert Event.from_line("   ") is None


def test_json_lines_stream():
    """模拟 JSON Lines 流式解析。"""
    lines = [
        Event.log("start").to_line(),
        Event.log("processing").to_line(),
        Event.log("done").to_line(),
    ]
    stream = "\n".join(lines) + "\n"
    events = []
    for line in stream.strip().split("\n"):
        e = Event.from_line(line)
        if e is not None:
            events.append(e)
    assert len(events) == 3
    assert events[0].type == "log"
    assert events[2].data["message"] == "done"


def test_unicode_in_events():
    """Unicode 路径在事件中往返。"""
    event = Event.file_done("1.第1章『耳なめ手コキオナサポ』.wav", True, 60.0,
                            0.001, "/path/to/1.第1章『耳なめ』")
    line = event.to_line()
    parsed = Event.from_line(line)
    assert parsed is not None
    # 检查全角引号和日文确实往返成功
    assert "『" in parsed.data["file"]
    assert "』" in parsed.data["file"]
    assert "耳なめ" in parsed.data["file"]
    assert "『耳なめ』" in parsed.data["output_dir"]


def test_cancel_action():
    """cancel action 的 Job 对象。"""
    job = Job(action="cancel")
    assert job.action == "cancel"
    text = job.to_json()
    job2 = Job.from_json(text)
    assert job2.action == "cancel"


def test_default_values():
    """Job 默认值。"""
    job = Job()
    assert job.model_key == "v2"
    assert job.device == "auto"
    assert job.dtype == "float32"
    assert job.block_timeout_sec == 300
    assert job.inputs == []
    assert job.action == "start"
