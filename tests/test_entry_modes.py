# -*- coding: utf-8 -*-
"""入口模式与冻结态兼容的契约测试（PLAN §2.3 / M7 打包前置改动）。

覆盖五处改动：
1. ``ui/app.py`` 的 ``--version`` / ``--worker`` 参数分派
2. ``core/batch.py`` 的 worker 启动命令（冻结态 vs 开发态）
3. ``paths.py`` 探测链第④路「从可执行文件向上」
4. ``.gitignore`` 不忽略 ``packaging/*.spec``
5. ``core/audio_io.py`` 的 ffmpeg 权限位自愈

★ 这些测试不 import torch —— GUI 进程禁止 import torch（PLAN §2.3）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, "src")

from audio_separation import paths as audio_paths
from audio_separation.core import audio_io, batch, env_check
from audio_separation.core.protocol import PROBE_FLAG, WORKER_FLAG

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestAppEntryModes:
    """``ui/app.py`` 的三种启动模式。"""

    def test_version_prints_and_exits_zero(self, capsys):
        from audio_separation.ui.app import main
        from audio_separation import __version__

        assert main(["--version"]) == 0
        assert capsys.readouterr().out.strip() == __version__

    def test_version_does_not_create_qapplication(self, monkeypatch, capsys):
        """★ `--version` 不能碰 QApplication —— 冻结态下它可能是无控制台的 exe。"""
        from audio_separation.ui import app

        def boom(*_args, **_kwargs):
            raise AssertionError("--version 分支不该创建 QApplication")

        monkeypatch.setattr("PySide6.QtWidgets.QApplication", boom)
        assert app.main(["--version"]) == 0
        capsys.readouterr()

    def test_worker_flag_is_dispatched_to_worker_main(self, monkeypatch):
        """`--worker` 必须转交给 worker 的 main，并把开关本身剥掉。"""
        from audio_separation.ui import app

        captured: list[list[str]] = []

        def fake_main(argv):
            captured.append(list(argv))
            return 0

        fake = types.ModuleType("audio_separation.worker")
        setattr(fake, "main", fake_main)
        monkeypatch.setitem(sys.modules, "audio_separation.worker", fake)

        assert app.main([WORKER_FLAG, "--job", "job.json"]) == 0
        assert captured == [["--job", "job.json"]], f"参数没剥干净：{captured}"

    def test_worker_mode_does_not_create_qapplication(self, monkeypatch):
        """★ worker 模式绝不能创建 QApplication（GUI 进程禁止 import torch）。"""
        from audio_separation.ui import app

        def boom(*_args, **_kwargs):
            raise AssertionError("worker 分支不该创建 QApplication")

        monkeypatch.setattr("PySide6.QtWidgets.QApplication", boom)

        fake = types.ModuleType("audio_separation.worker")
        setattr(fake, "main", lambda argv: 0)
        monkeypatch.setitem(sys.modules, "audio_separation.worker", fake)

        assert app.main([WORKER_FLAG, "--job", "job.json"]) == 0

    def test_argv_defaults_to_sys_argv(self, monkeypatch):
        """``main()`` 无参调用时读 sys.argv（entry_points 就是这么调的）。"""
        from audio_separation.ui import app
        from audio_separation import __version__

        monkeypatch.setattr(sys, "argv", ["audio_separation", "--version"])
        assert app.main() == 0


class TestWorkerCommand:
    """``core/batch.py`` 的 worker 启动命令。"""

    def test_frozen_uses_executable_with_worker_flag(self, monkeypatch, tmp_path):
        """★ 冻结态：worker.py 只是 _MEIPASS 里的源码文本，只能让 exe 自己重启。"""
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        exe = tmp_path / "audio_separation.exe"
        monkeypatch.setattr(audio_paths, "self_executable", lambda: exe)

        cmd, _env = batch._worker_command(Path("job.json"))

        assert cmd == [str(exe), WORKER_FLAG, "--job", "job.json"]

    def test_dev_uses_module_invocation(self, monkeypatch):
        """开发态走 `python -m audio_separation.worker`。"""
        monkeypatch.delattr(sys, "frozen", raising=False)

        cmd, _env = batch._worker_command(Path("job.json"))

        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "audio_separation.worker"]
        assert cmd[3:] == ["--job", "job.json"]

    def test_dev_mode_exports_package_root_on_pythonpath(self, monkeypatch):
        """子进程得能 import audio_separation —— src 只在当前进程的 sys.path 里。"""
        monkeypatch.delattr(sys, "frozen", raising=False)

        _cmd, env = batch._worker_command(Path("job.json"))

        assert "PYTHONPATH" in env
        entries = env["PYTHONPATH"].split(os.pathsep)
        assert str(REPO_ROOT / "src") in entries

    def test_run_worker_passes_computed_command(self, monkeypatch, tmp_path):
        """``run_worker`` 必须真的用上面算出来的命令（别各算各的）。"""
        monkeypatch.delattr(sys, "frozen", raising=False)
        seen: dict[str, Any] = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                seen["cmd"] = cmd
                seen["kwargs"] = kwargs

        monkeypatch.setattr(batch.subprocess, "Popen", _FakePopen)

        batch.run_worker(tmp_path / "job.json")

        cmd = seen["cmd"]
        assert cmd[1:3] == ["-m", "audio_separation.worker"]
        assert "PYTHONPATH" in seen["kwargs"]["env"]


class TestRepoRootProbeChain:
    """``paths.py`` 探测链新增的第④路。"""

    def test_diagnose_lists_executable_probe(self):
        diag = audio_paths.diagnose()
        assert "④ 从可执行文件" in diag, f"探测链少了第④路：\n{diag}"

    def test_executable_probe_can_resolve_root(self, monkeypatch, tmp_path):
        """★ cwd 与源码位置都落空时，第④路能靠 exe 旁边的标记命中仓库根。

        这就是双击 exe 的真实场景：cwd 在 C:\\Windows\\System32。
        """
        fake_root = tmp_path / "dist"
        (fake_root / "models").mkdir(parents=True)
        (fake_root / "ffmpeg").mkdir()
        exe = fake_root / "audio_separation.exe"
        exe.write_text("", encoding="utf-8")

        def fake_probe(start: Path):
            if str(start) == str(exe):
                return fake_root, [f"  ✔ {fake_root}"]
            return None, [f"  ✗ {start}   （测试桩）"]

        monkeypatch.setattr(audio_paths, "self_executable", lambda: exe)
        monkeypatch.setattr(audio_paths, "_probe", fake_probe)
        monkeypatch.delenv(audio_paths.ENV_ROOT, raising=False)
        audio_paths.reset_cache()
        try:
            assert audio_paths.repo_root() == fake_root
        finally:
            audio_paths.reset_cache()


class TestFfmpegPermission:
    """``audio_io._ensure_executable``。"""

    def test_windows_returns_path_unchanged(self, tmp_path):
        exe = tmp_path / "ffmpeg.exe"
        exe.write_text("", encoding="utf-8")
        assert audio_io._ensure_executable(exe) == exe

    @pytest.mark.skipif(sys.platform == "win32", reason="权限位在 Windows 上无意义")
    def test_linux_adds_execute_bit(self, tmp_path):
        """★ PyInstaller --add-binary 会丢 +x，打进 bundle 的 ffmpeg 跑不了。"""
        exe = tmp_path / "ffmpeg"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        exe.chmod(0o644)
        assert not os.access(exe, os.X_OK)

        audio_io._ensure_executable(exe)

        assert os.access(exe, os.X_OK)

    @pytest.mark.skipif(sys.platform == "win32", reason="权限位在 Windows 上无意义")
    def test_linux_keeps_existing_execute_bit(self, tmp_path):
        exe = tmp_path / "ffmpeg"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        exe.chmod(0o755)
        before = exe.stat().st_mode

        audio_io._ensure_executable(exe)

        assert exe.stat().st_mode == before


class TestGitignoreKeepsSpec:
    """``packaging/*.spec`` 是要提交的源码，不能被 ``*.spec`` 规则吞掉。"""

    def test_negation_rule_present(self):
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "!packaging/*.spec" in text

    def test_spec_is_not_ignored_by_git(self):
        try:
            proc = subprocess.run(
                ["git", "check-ignore", "-q", "packaging/audio_separation.spec"],
                cwd=REPO_ROOT,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            pytest.skip("git 不可用")
        if proc.returncode == 128:
            pytest.skip("不在 git 仓库里")
        assert proc.returncode != 0, "packaging/*.spec 被 .gitignore 忽略了，spec 提交不上去"


class TestProbeEntryMode:
    """``ui/app.py`` 的 ``--probe`` 分派（冻结态设备探测）。"""

    def test_probe_flag_is_dispatched_to_probe_main(self, monkeypatch):
        """``--probe`` 必须转交给 device_probe 的 main。"""
        from audio_separation.ui import app

        captured: list[bool] = []

        def fake_main():
            captured.append(True)
            return 0

        fake = types.ModuleType("audio_separation.core.device_probe")
        setattr(fake, "main", fake_main)
        monkeypatch.setitem(sys.modules, "audio_separation.core.device_probe", fake)

        assert app.main([PROBE_FLAG]) == 0
        assert captured == [True]

    def test_probe_mode_does_not_create_qapplication(self, monkeypatch):
        """★ 探测模式绝不能创建 QApplication（GUI 进程禁止 import torch）。"""
        from audio_separation.ui import app

        def boom(*_args, **_kwargs):
            raise AssertionError("探测分支不该创建 QApplication")

        monkeypatch.setattr("PySide6.QtWidgets.QApplication", boom)

        fake = types.ModuleType("audio_separation.core.device_probe")
        setattr(fake, "main", lambda: 0)
        monkeypatch.setitem(sys.modules, "audio_separation.core.device_probe", fake)

        assert app.main([PROBE_FLAG]) == 0


class TestProbeCommand:
    """``core/env_check.py`` 启动探测子进程的命令。"""

    @staticmethod
    def _capture_cmd(monkeypatch) -> list[str]:
        seen: dict[str, Any] = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                seen["cmd"] = cmd

            def communicate(self, timeout=None):
                raise RuntimeError("测试到此为止")

            def kill(self):
                pass

        monkeypatch.setattr(env_check.subprocess, "Popen", _FakePopen)
        env_check.probe_device(force=True)
        return list(seen["cmd"])

    def test_frozen_uses_executable_with_probe_flag(self, monkeypatch):
        """★ 冻结态：没有 `python -c` 这种跑法，只能让 exe 自己认 --probe 重启。"""
        monkeypatch.setattr(sys, "frozen", True, raising=False)

        cmd = self._capture_cmd(monkeypatch)

        assert cmd == [sys.executable, PROBE_FLAG]

    def test_dev_runs_probe_script_file(self, monkeypatch):
        """开发态直接跑脚本文件（不用 -m：-m 会把 cwd 塞进 sys.path[0]）。"""
        monkeypatch.delattr(sys, "frozen", raising=False)

        cmd = self._capture_cmd(monkeypatch)

        assert cmd[0] == sys.executable
        assert cmd[1].endswith("device_probe.py"), f"探测脚本路径不对：{cmd[1]}"

    def test_probe_script_is_self_contained(self):
        """★ device_probe 不能 import 项目内模块 —— 它要以独立脚本方式运行。"""
        src = (REPO_ROOT / "src" / "audio_separation" / "core" / "device_probe.py").read_text(
            encoding="utf-8"
        )
        assert "import audio_separation" not in src, "探测脚本必须自包含"
        assert "from ." not in src, "探测脚本不能用相对 import"
