# -*- coding: utf-8 -*-
"""日志落盘与控制台隐藏的契约测试。

★这里断言的是**契约**（文件落在哪、内容进不进、切完会不会丢），
不是某个具体实现细节 —— 换一种 tee 实现方式测试就该跟着改，那样的测试没价值。
"""

import logging
import sys
from pathlib import Path

import pytest

from audio_separation.core import logging_setup
from audio_separation.core.console_hider import hide_console
from audio_separation.core.logging_setup import (
    _TeeStream,
    log_file_path,
    log_line,
    logs_dir,
    setup_console_logging,
)


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """把日志目录指到临时目录，并把模块状态复位。

    ★必须复位 `_installed` / `_log_file` / `_tees` ——
    否则一个测试装上 tee 流之后，后面的测试全被 `_installed` 挡住。
    """
    monkeypatch.setattr(logging_setup, "logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(logging_setup, "_installed", False)
    monkeypatch.setattr(logging_setup, "_log_file", None)
    monkeypatch.setattr(logging_setup, "_tees", [])
    yield tmp_path / "logs"
    # 还原 stdout/stderr 与 root logger，别污染别的测试
    for tee in getattr(logging_setup, "_tees", []):
        pass
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_audio_separation_test", False):
            root.removeHandler(handler)


class TestLogsDir:
    def test_points_to_logs_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "audio_separation.paths.repo_root", lambda: tmp_path
        )
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert logging_setup.logs_dir() == tmp_path / "logs"

    def test_frozen_uses_exe_parent(self, tmp_path, monkeypatch):
        """冻结态：日志跟着 exe 走，而不是跟着当前工作目录。"""
        import audio_separation.paths as paths

        exe = tmp_path / "app" / "audio_separation.exe"
        monkeypatch.setattr(paths, "self_executable", lambda: exe)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert paths.logs_dir() == exe.parent / "logs"

    def test_log_file_name_has_date(self, isolated_logging):
        path = log_file_path()
        assert path.parent == isolated_logging
        assert path.name.startswith("app-")
        assert path.suffix == ".log"


class TestTeeStream:
    def test_writes_to_both_streams(self):
        import io

        original = io.StringIO()
        sink = io.StringIO()
        tee = _TeeStream(original, sink)
        tee.write("hello")
        tee.flush()
        assert original.getvalue() == "hello"
        assert sink.getvalue() == "hello"

    def test_survives_dead_original(self):
        """原控制台句柄失效时不能抛异常 —— 日志写不进去不该让程序崩。"""
        import io

        sink = io.StringIO()

        class _Dead(io.StringIO):
            def write(self, s):
                raise OSError("句柄已失效")

        tee = _TeeStream(_Dead(), sink)
        tee.write("still works")
        tee.flush()
        assert sink.getvalue() == "still works"

    def test_detach_original_stops_console_writes(self):
        import io

        original = io.StringIO()
        sink = io.StringIO()
        tee = _TeeStream(original, sink)
        tee.detach_original()
        tee.write("only to file")
        tee.flush()
        assert original.getvalue() == ""
        assert sink.getvalue() == "only to file"

    def test_isatty_is_false(self):
        import io

        tee = _TeeStream(io.StringIO(), io.StringIO())
        assert tee.isatty() is False
        assert tee.writable() is True

    def test_encoding_property_never_none(self):
        import io

        tee = _TeeStream(None, io.StringIO())
        assert tee.encoding == "utf-8"


class TestSetupConsoleLogging:
    def test_creates_log_dir_and_file(self, isolated_logging):
        path = setup_console_logging()
        assert path is not None
        assert path.exists()
        assert path.parent == isolated_logging

    def test_print_goes_to_log_file(self, isolated_logging):
        """★核心契约：print() 的内容必须落进日志文件。"""
        path = setup_console_logging()
        print("这是一条测试日志")
        sys.stdout.flush()
        content = path.read_text(encoding="utf-8")
        assert "这是一条测试日志" in content

    def test_logging_call_goes_to_same_file(self, isolated_logging):
        path = setup_console_logging()
        log_line("业务日志行")
        sys.stdout.flush()
        content = path.read_text(encoding="utf-8")
        assert "业务日志行" in content

    def test_stderr_also_captured(self, isolated_logging):
        path = setup_console_logging()
        print("错误信息", file=sys.stderr)
        sys.stderr.flush()
        assert "错误信息" in path.read_text(encoding="utf-8")

    def test_returns_none_when_dir_uncreatable(self, monkeypatch, tmp_path):
        """打不开日志目录时返回 None，而不是抛异常。"""
        monkeypatch.setattr(
            logging_setup, "logs_dir", lambda: tmp_path / "logs"
        )
        monkeypatch.setattr(logging_setup, "_installed", False)
        monkeypatch.setattr(logging_setup, "_log_file", None)

        real_mkdir = Path.mkdir

        def _boom(self, *args, **kwargs):
            raise OSError("没权限")

        monkeypatch.setattr(Path, "mkdir", _boom)
        assert setup_console_logging() is None
        monkeypatch.setattr(Path, "mkdir", real_mkdir)

    def test_idempotent(self, isolated_logging):
        first = setup_console_logging()
        second = setup_console_logging()
        assert first == second


class TestLogSizeCap:
    """★日志文件必须有体积上限。

    日志内容来自「程序自己打到 stdout 的东西」，而 stdout 的内容有可能
    反过来包含日志文件自身（脚本读日志再打印，tee 又把打印内容写回日志），
    形成自我放大的正反馈。实测这个回路能把单个文件撑到 50 GB。
    """

    def test_rotates_when_exceeding_limit(self, tmp_path):
        path = tmp_path / "app.log"
        sink = logging_setup._Sink(path, max_bytes=200)
        for index in range(50):
            sink.write(f"第 {index} 行日志，填充内容填充内容填充内容\n")
        sink.flush()
        assert path.stat().st_size <= 400
        assert path.with_suffix(".1.log").exists()

    def test_keeps_writing_after_rotation(self, tmp_path):
        """轮转后还得能继续写 —— 否则后半程日志全丢。"""
        path = tmp_path / "app.log"
        sink = logging_setup._Sink(path, max_bytes=200)
        for index in range(30):
            sink.write(f"行 {index}\n")
        sink.write("轮转之后的关键一行\n")
        sink.flush()
        sink.close()
        assert "轮转之后的关键一行" in path.read_text(encoding="utf-8")

    def test_never_exceeds_limit_by_more_than_one_write(self, tmp_path):
        path = tmp_path / "app.log"
        sink = logging_setup._Sink(path, max_bytes=100)
        for _ in range(200):
            sink.write("x" * 30 + "\n")
        sink.flush()
        assert path.stat().st_size < 100 + 40

    def test_normal_usage_does_not_rotate(self, tmp_path):
        """正常日志量不该触发轮转（否则每次启动都留个 .1.log 垃圾）。"""
        path = tmp_path / "app.log"
        sink = logging_setup._Sink(path, max_bytes=10 * 1024 * 1024)
        sink.write("普通的一行日志\n")
        sink.flush()
        assert not path.with_suffix(".1.log").exists()


class TestHideConsole:
    def test_returns_bool_without_raising(self):
        """隐藏失败绝不能抛异常。"""
        result = hide_console()
        assert isinstance(result, bool)

    def test_non_windows_returns_false(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert hide_console() is False
