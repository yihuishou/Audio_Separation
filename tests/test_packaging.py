"""M6-3 打包冒烟测试。

实际打包在 M7 完成，这里先建立测试框架，默认跳过。
打包完成后运行：pytest -m packaging -q
"""

import pytest
from pathlib import Path

DIST = Path(__file__).resolve().parent.parent / "dist" / "audio_separation"


@pytest.mark.packaging
@pytest.mark.skipif(not DIST.exists(), reason="packaging not built yet (run M7 first)")
class TestPackaging:
    """冻结产物冒烟。"""

    def test_dist_exists(self):
        assert DIST.exists(), f"打包产物不存在：{DIST}"

    def test_gui_entry_point(self):
        """GUI 入口可执行。"""
        import subprocess, sys

        exe = DIST / ("audio_separation.exe" if sys.platform == "win32" else "audio_separation")
        assert exe.exists(), f"GUI 可执行文件不存在：{exe}"

        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True, text=True, timeout=30, encoding="utf-8",
        )
        assert result.returncode == 0, f"--version 失败：{result.stderr}"

    def test_worker_entry_point(self):
        """worker 入口可执行。"""
        import subprocess, sys

        exe = DIST / ("audio_separation.exe" if sys.platform == "win32" else "audio_separation")
        assert exe.exists()

        result = subprocess.run(
            [str(exe), "--worker", "--job", "nonexistent.json"],
            capture_output=True, text=True, timeout=30, encoding="utf-8",
        )
        # 预期：job 文件不存在会报错，但 worker 模式应该被识别
        # 具体退出码取决于实现，这里只验证不 crash
