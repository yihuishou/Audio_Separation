# -*- coding: utf-8 -*-
"""worker ↔ GUI 之间的 JSON Lines 协议（PLAN §2.3 / §4 M3-1）。

为什么是 JSON Lines（一行一个 JSON 对象）而不是一个大的 JSON 数组：
- **流式**：worker 每产生一条事件就写一行，GUI 边收边解析，不必等 worker
  跑完整个队列才拿到第一条进度；
- **故障恢复**：中途崩掉时，已写入的行都是完整的（每行以换行分隔），
  GUI 可以解析到崩溃前的最后一条事件；
- **行缓冲友好**：`print(json.dumps(...), flush=True)` 天然产生这种格式。

## 事件类型（6 种）

| type         | 含义 | 关键字段 |
|--------------|------|----------|
| `log`        | 人读日志行 | `message` |
| `progress`   | 单文件分块进度 | `file`, `block`, `total_blocks`, `eta_sec` |
| `file_done`  | 单个文件完成 | `file`, `ok`, `elapsed_sec`, `mix_error_pct`, `output_dir` |
| `fatal`      | 无法恢复的错误 | `message` |
| `cancelled`  | 用户/看门狗取消 | `reason` |
| `done`       | 整个队列完成 | `total`, `ok`, `failed`, `cancelled` |

## job.json schema

```json
{
  "version": 1,
  "model_key": "v2",
  "device": "auto",
  "dtype": "float32",
  "block_timeout_sec": 300,
  "inputs": [
    {"src": "/path/to/input.wav", "output_dir": "/path/to/output"}
  ]
}
```

★ **worker 进程绝不 import torch/bs_roformer** —— 它只做协议编解码、
JSON 读写、子进程管理与信号处理。推理在另一个子进程里跑（由 `worker.py`
spawn，见 PLAN §2.3）。这样 torch 的 1.5 s import 不会拖慢 GUI 启动，
也不会在取消时卡死 worker。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

JobAction = Literal["start", "cancel"]

#: worker 模式的命令行开关。★冻结态下 exe 自己就是 worker（PLAN §2.3 / M7），
#: GUI 入口（`ui/app.py`）与调度器（`core/batch.py`）必须用同一个字符串。
WORKER_FLAG = "--worker"


@dataclass
class JobInput:
    """一个待分离的文件。"""
    src: str
    output_dir: str | None = None


@dataclass
class Job:
    """job.json 的内存表示（PLAN §4 M3-1）。"""
    version: int = 1
    model_key: str = "v2"
    device: str = "auto"
    dtype: str = "float32"
    block_timeout_sec: int = 300
    inputs: list[JobInput] = field(default_factory=list)
    action: JobAction = "start"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Job:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Job:
        inputs = [
            JobInput(src=i["src"], output_dir=i.get("output_dir"))
            for i in d.get("inputs", [])
        ]
        return cls(
            version=d.get("version", 1),
            model_key=d.get("model_key", "v2"),
            device=d.get("device", "auto"),
            dtype=d.get("dtype", "float32"),
            block_timeout_sec=d.get("block_timeout_sec", 300),
            inputs=inputs,
            action=d.get("action", "start"),
        )


EventType = Literal["log", "progress", "file_done", "fatal", "cancelled", "done"]


@dataclass
class Event:
    """一条 JSON Lines 事件。"""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        """序列化为一行 JSON（不带换行，由调用方负责 flush 与换行）。"""
        return json.dumps({"type": self.type, "data": self.data}, ensure_ascii=False)

    @classmethod
    def from_line(cls, line: str) -> Event | None:
        line = line.strip()
        if not line:
            return None
        obj = json.loads(line)
        return cls(type=obj["type"], data=obj.get("data", {}))

    # ── 构造函数 ──────────────────────────────────────────────────────────
    @staticmethod
    def log(message: str) -> Event:
        return Event(type="log", data={"message": message})

    @staticmethod
    def progress(file: str, block: int, total_blocks: int, eta_sec: float) -> Event:
        return Event(
            type="progress",
            data={"file": file, "block": block, "total_blocks": total_blocks,
                  "eta_sec": round(eta_sec, 1)},
        )

    @staticmethod
    def file_done(
        file: str, ok: bool, elapsed_sec: float, mix_error_pct: float | None,
        output_dir: str,
    ) -> Event:
        return Event(
            type="file_done",
            data={"file": file, "ok": ok, "elapsed_sec": round(elapsed_sec, 1),
                  "mix_error_pct": mix_error_pct, "output_dir": output_dir},
        )

    @staticmethod
    def fatal(message: str) -> Event:
        return Event(type="fatal", data={"message": message})

    @staticmethod
    def cancelled(reason: str) -> Event:
        return Event(type="cancelled", data={"reason": reason})

    @staticmethod
    def done(total: int, ok: int, failed: int, cancelled: int) -> Event:
        return Event(
            type="done",
            data={"total": total, "ok": ok, "failed": failed, "cancelled": cancelled},
        )


# ── 验证 ──────────────────────────────────────────────────────────────────
EVENT_FIELDS: dict[EventType, list[str]] = {
    "log": ["message"],
    "progress": ["file", "block", "total_blocks"],
    "file_done": ["file", "ok", "elapsed_sec", "output_dir"],
    "fatal": ["message"],
    "cancelled": ["reason"],
    "done": ["total", "ok", "failed", "cancelled"],
}


def validate_event(event: Event) -> list[str]:
    """返回缺失字段列表；空列表表示合法。"""
    required = EVENT_FIELDS.get(event.type, [])
    missing = [f for f in required if f not in event.data]
    if event.type == "progress" and "block" in event.data:
        if not isinstance(event.data["block"], int) or event.data["block"] < 1:
            missing.append("block (must be int >= 1)")
    if event.type == "file_done" and "ok" in event.data:
        if not isinstance(event.data["ok"], bool):
            missing.append("ok (must be bool)")
    return missing
