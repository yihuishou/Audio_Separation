# -*- coding: utf-8 -*-
"""后台批处理 worker（PLAN §4 M4「待完善：ui/workers.py（后台 worker 管理）」）。

为什么需要它
------------
推理跑在**独立子进程**里（`worker.py`），GUI 通过 JSON Lines 收事件。
但 GUI 线程仍然不能阻塞在读子进程 stdout 上 —— 那样窗口会变白、菜单点不动、
「取消」按钮按下去没反应。所以这里用 `QThread` 把 `core.batch.run_batch`
的事件流搬到后台线程，再用 Qt 信号搬回主线程刷新界面。

★ 本模块**不 import torch**：GUI 进程始终不加载重依赖（PLAN §2.3）。
  `core.batch` 也不 import torch —— 它只负责起子进程、编解码协议。
  真正碰 torch 的是 `worker.py` 那条子进程。

线程与取消的时序
----------------
取消靠哨兵文件（`job_dir/.cancel`）。有个坑：worker 启动时会**主动删掉**
上一次遗留的哨兵（`worker.py:82-84`），所以只在点「取消」的那一刻写一次哨兵
是可能丢失的 —— 若那一刻 worker 还没启动，哨兵会被它删掉，取消就失效了。

对策：worker 线程每收到一条事件就检查一次取消标志，为真则**补写**哨兵。
第一条事件（`resolving device`）一定发生在 worker 启动之后，所以补写一定
落在「启动删除」之后，哨兵必定被 worker 看见。
"""
from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..core.batch import cancel_job, run_batch


class BatchWorker(QThread):
    """在后台线程里跑批量分离，把事件流搬回 GUI 线程。

    信号：
        log_line(str)            —— 一行人读日志
        progress_updated(int,int,str,int)
                                   —— (已完成文件数, 文件总数, 附加说明, 百分比)
        file_finished(str,bool,str)
                                   —— (文件名, 是否成功, 输出目录)
        fatal_error(str)         —— 无法恢复的错误
        run_cancelled(str)       —— 取消原因
        batch_finished(dict)     —— 汇总 {total, ok, failed, cancelled}
    """

    log_line = Signal(str)
    progress_updated = Signal(int, int, str, int)
    file_finished = Signal(str, bool, str)
    fatal_error = Signal(str)
    run_cancelled = Signal(str)
    batch_finished = Signal(dict)

    def __init__(
        self,
        inputs: list[dict[str, str]],
        *,
        model_key: str = "v2",
        device: str = "auto",
        dtype: str = "float32",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._inputs = list(inputs)
        self._model_key = model_key
        self._device = device
        self._dtype = dtype
        # 作业目录自己建：job.json 与取消哨兵都落在这里，
        # `core.batch.run_batch` 拿到它才能写哨兵。
        self._job_dir = Path(tempfile.mkdtemp(prefix="audio_sep_gui_job_"))
        self._cancel_requested = False
        self._lock = threading.Lock()

    # ── 取消 ──────────────────────────────────────────────────────────────

    def request_cancel(self) -> None:
        """请求取消。可从 GUI 主线程直接调用。"""
        with self._lock:
            self._cancel_requested = True
        # 先写一次：worker 若已经在跑，分块间就能看到。
        self._write_sentinel()

    @property
    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def _write_sentinel(self) -> None:
        """写取消哨兵（失败不致命 —— 只是这次取消没生效而已）。"""
        try:
            cancel_job(self._job_dir)
        except OSError:
            pass

    # ── 主循环 ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """QThread 主体：消费事件流并转成 Qt 信号。"""
        total = len(self._inputs)
        completed = 0
        summary: dict[str, int] = {
            "total": total, "ok": 0, "failed": 0, "cancelled": 0,
        }
        try:
            for event in self._iter_events():
                # 补写哨兵：worker 启动时会删旧哨兵，只在点取消时写一次可能丢。
                if self.cancel_requested:
                    self._write_sentinel()

                kind = event.type
                data = event.data

                if kind == "log":
                    self.log_line.emit(str(data.get("message", "")))

                elif kind == "progress":
                    block = int(data.get("block", 0) or 0)
                    blocks = int(data.get("total_blocks", 0) or 0) or 1
                    eta = float(data.get("eta_sec", 0.0) or 0.0)
                    frac = min(1.0, max(0.0, block / blocks))
                    pct = int(100 * (completed + frac) / total) if total else 0
                    name = str(data.get("file", ""))
                    msg = f"{name}  {block}/{blocks} 块，剩余约 {eta:.0f}s"
                    self.progress_updated.emit(completed, total, msg, pct)

                elif kind == "file_done":
                    completed += 1
                    ok = bool(data.get("ok", False))
                    name = str(data.get("file", ""))
                    out_dir = str(data.get("output_dir", ""))
                    self.file_finished.emit(name, ok, out_dir)
                    pct = int(100 * completed / total) if total else 0
                    self.progress_updated.emit(completed, total, "", pct)

                elif kind == "fatal":
                    self.fatal_error.emit(str(data.get("message", "")))

                elif kind == "cancelled":
                    self.run_cancelled.emit(str(data.get("reason", "")))

                elif kind == "done":
                    summary = {
                        "total": int(data.get("total", total)),
                        "ok": int(data.get("ok", 0)),
                        "failed": int(data.get("failed", 0)),
                        "cancelled": int(data.get("cancelled", 0)),
                    }

            self.batch_finished.emit(summary)

        except Exception as exc:  # 兜底：线程里抛异常不能静默吞掉
            self.fatal_error.emit(f"批处理异常：{exc}")
        finally:
            shutil.rmtree(self._job_dir, ignore_errors=True)

    def _iter_events(self):
        """产出 `core.batch` 的事件流。"""
        return run_batch(
            self._inputs,
            model_key=self._model_key,
            device=self._device,
            dtype=self._dtype,
            job_dir=self._job_dir,
        )
