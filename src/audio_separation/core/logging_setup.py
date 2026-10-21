# -*- coding: utf-8 -*-
"""控制台日志落盘（GUI 模式专用）。

## 为什么需要它

GUI 模式要把控制台窗口藏起来（见 `console_hider.py`），但程序里到处是
`print()` —— 直接藏掉窗口等于把这些输出全丢了，出问题时用户只能干瞪眼。

所以这里做两件事：

1. 把 `sys.stdout` / `sys.stderr` 换成** tee 流**：一边写原处（开发态还能
   在终端看到），一边写日志文件。
2. 额外挂一个 `logging` 的文件处理器，让 `logging.info(...)` 之类的调用
   也进同一个文件。

## 日志文件的位置

`<exe 同级目录>/logs/app-<日期>.log`。按天切分，避免单个文件无限膨胀。

★冻结态下 "exe 同级目录" 取 `sys.executable` 的父目录；开发态取仓库根
（`paths.repo_root()`）—— 开发时不该往 Python 安装目录里写东西，
那里通常没有写权限。

## ★GUI 模式下 `print()` 本来是被丢掉的

打包用 `console=False`（GUI 子系统），这种程序**不分配控制台**，
`sys.stdout` 是 `None`。而 `print()` 遇到 `None` 的 stdout 是**静默 no-op**
—— 也就是说 GUI 模式里所有 `print()` 原本都凭空消失，什么都留不下。

装上 tee 之后 stdout 变成真实对象，print 的内容这才终于能落进日志文件。
所以这个模块不只是"锦上添花"，它是 GUI 模式下唯一的输出出口。

## ★worker 模式绝不能调用这里

worker 的 stdout 是 JSON Lines 协议通道（`core/protocol.py`），
调度器靠 `core/batch.py` 的 `stdout=PIPE` 逐行解析。任何往 stdout 写的
额外内容都会让协议解析失败。所以 `setup_console_logging()` 只在 GUI 分支调用。
"""
from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

#: 日志文件名里用的格式
_STAMP = "%Y%m%d"

#: 保留最近多少天的日志文件
_KEEP_DAYS = 14

#: 单个日志文件的大小上限（字节）。超过就轮转。
_MAX_BYTES = 10 * 1024 * 1024

#: tee 流已经装过没有 —— 重复调用会套娃
_installed = False

#: 当前生效的日志文件路径（setup 之前是 None）
_log_file: Path | None = None

#: 已安装的 tee 流，用于事后摘掉「原控制台」那一路
_tees: list[_TeeStream] = []

#: 业务日志用的 logger 名字
_LOGGER_NAME = "audio_separation"


def logs_dir() -> Path:
    """日志目录：exe 同级目录下的 `logs/`（开发态是仓库根下的 `logs/`）。

    目录**不创建** —— 纯路径函数不该有副作用，调用点自己建。
    """
    from .. import paths

    return paths.logs_dir()


def log_file_path() -> Path:
    """当前日志文件的完整路径（不创建文件）。"""
    stamp = datetime.now().strftime(_STAMP)
    return logs_dir() / f"app-{stamp}.log"


class _Sink:
    """日志文件写入端，带体积上限与轮转。

    ★**为什么必须有上限**：日志记录的是「程序自己打到 stdout 的内容」，
    而 stdout 的内容可能反过来包含日志文件自身（脚本读日志再打印，
    tee 又把打印内容写回日志），形成自我放大的正反馈。
    实测这个回路能把单个日志文件撑到 50 GB 把磁盘写满。
    上限一到就轮转成 `.1.log`，无论多 pathological 的输出都撑不爆磁盘。
    """

    def __init__(self, path: Path, max_bytes: int = _MAX_BYTES) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._handle = self._open()

    def _open(self) -> TextIO:
        return open(
            self._path, "a", encoding="utf-8", errors="replace", buffering=1
        )

    def _rotate(self) -> None:
        try:
            self._handle.close()
        except (ValueError, OSError):
            pass
        backup = self._path.with_suffix(".1.log")
        try:
            if backup.exists():
                backup.unlink()
            self._path.replace(backup)
        except OSError:
            pass
        self._handle = self._open()

    def write(self, s: str) -> int:
        try:
            if self._handle.tell() >= self._max_bytes:
                self._rotate()
        except (ValueError, OSError):
            pass
        try:
            self._handle.write(s)
        except (ValueError, OSError):
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._handle.flush()
        except (ValueError, OSError):
            pass

    def close(self) -> None:
        try:
            self._handle.close()
        except (ValueError, OSError):
            pass

    def fileno(self) -> int:
        return self._handle.fileno()


class _TeeStream:
    """把写入同时送到两个流：原流 + 日志文件。

    ★**故意不继承 `io.TextIOBase`** —— 它的 `encoding` / `errors` 是 C 层
    只读属性（`AttributeError: attribute 'encoding' of '_io._TextIOBase'
    objects is not writable`），子类用 property 覆盖会被 Pyright 判为
    不兼容的变量覆盖。这里自己实现需要的最小接口。

    ★原流可能已经失效（控制台被 `FreeConsole()` 摘掉、或冻结态本来就没有
    控制台）。写入失效句柄会抛 `OSError`，这里吞掉 —— 日志写不进去不该
    让程序崩。
    """

    def __init__(self, original: TextIO | None, sink: "_Sink") -> None:
        self._original = original
        self._sink = sink
        self.encoding = getattr(original, "encoding", None) or "utf-8"
        self.errors = getattr(original, "errors", None) or "replace"
        self.closed = False

    def detach_original(self) -> None:
        """不再往原控制台写（只留日志文件那一路）。"""
        self._original = None

    def write(self, s: str) -> int:
        if self._original is not None:
            try:
                self._original.write(s)
            except (ValueError, OSError):
                # 原流已关闭（比如控制台被 FreeConsole 摘掉）
                pass
        try:
            self._sink.write(s)
        except (ValueError, OSError):
            pass
        return len(s)

    def writelines(self, lines: list[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        for stream in (self._original, self._sink):
            if stream is None:
                continue
            try:
                stream.flush()
            except (ValueError, OSError):
                pass

    def fileno(self) -> int:
        if self._original is None:
            raise io.UnsupportedOperation("fileno")
        return self._original.fileno()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        self.flush()
        self.closed = True


def _cleanup_old_logs(directory: Path) -> None:
    """删掉超过保留期的旧日志文件。失败不影响主流程。"""
    try:
        cutoff = datetime.now().timestamp() - _KEEP_DAYS * 86400
        for path in directory.glob("app-*.log"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
    except OSError:
        pass


def setup_console_logging(verbose: bool = False) -> Path | None:
    """把 stdout/stderr 与 logging 都导向日志文件。

    Args:
        verbose: 是否把日志级别降到 DEBUG（默认 INFO）。

    Returns:
        日志文件路径。**打不开就返回 None** —— 日志是辅助功能，
        绝不能因为写不了日志就让整个程序起不来。
    """
    global _installed, _log_file

    if _installed:
        return _log_file

    directory = logs_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    path = log_file_path()
    try:
        # ★buffering=1（行缓冲）—— 默认的块缓冲（8 KB）意味着日志要攒够
        #   8 KB 或进程正常退出才落盘。程序被强杀 / 崩溃时缓冲区里的日志
        #   全丢了，而那恰恰是最需要日志的时候。
        sink = _Sink(path)
    except OSError:
        return None

    atexit_register(sink)

    for stream in (sys.stdout, sys.stderr):
        tee = _TeeStream(stream, sink)
        _tees.append(tee)
    sys.stdout, sys.stderr = _tees[0], _tees[1]

    # ★用 StreamHandler 而不是 FileHandler —— 后者会另开一个句柄写同一个文件，
    #   两个独立句柄在 Windows 上会交错写乱；共用 sink 这一个句柄才不会。
    handler = logging.StreamHandler(sink)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(handler)

    _log_file = path
    _installed = True
    _cleanup_old_logs(directory)
    return path


def detach_console_output() -> None:
    """停止往原控制台写，只留日志文件。

    控制台窗口被 `FreeConsole()` 摘掉之后，原来的句柄就失效了。
    与其每次写都撞一次异常，不如直接断掉这一路。
    """
    for tee in _tees:
        tee.detach_original()


def log_line(message: str) -> None:
    """把一条业务日志写进日志文件。

    GUI 日志面板里的内容（worker 的输出、完成/失败/取消提示）都走这里，
    这样用户看到的和文件里记的是同一份。
    """
    logging.getLogger(_LOGGER_NAME).info(message)


def atexit_register(sink: "_Sink") -> None:
    """注册退出时关闭日志文件的回调。

    拆成独立函数是为了让测试能 monkeypatch 掉（atexit 直接注册的话，
    测试之间会互相干扰）。
    """
    import atexit

    def _close() -> None:
        try:
            sink.flush()
            sink.close()
        except (ValueError, OSError):
            pass

    atexit.register(_close)


def current_log_file() -> Path | None:
    """当前生效的日志文件路径（没 setup 过就是 None）。"""
    return _log_file
