"""PLAN §2.8 启动期规避的契约测试。

这两条规避的失败模式是**静默死循环** —— 没有异常、没有日志、进程只是永远不返回。
实测：没有 `NUMBA_CACHE_DIR` 时 `python -m audio_separation.core.cli -i <音频>`
会在 `from bs_roformer import BSRoformer` 里挂 50 分钟以上，栈顶固定是
``tempfile._mkstemp_inner`` ← ``numba.core.caching.ensure_cache_path``
← ``librosa/core/notation.py:1030 <module>``。

所以这里不能只测"函数存在"，必须测**在干净子进程里、入口不额外做任何准备**
时也能跑通 —— 那才是生产入口的真实形状。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def _run(script: str, *, timeout: int = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # ★故意不传 NUMBA_CACHE_DIR：要验证的是包入口自己会设，
    #   而不是继承了一个恰好正确的环境。
    env.pop("NUMBA_CACHE_DIR", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        cwd=str(SRC.parent), env=env,
    )


class TestNumbaCacheGuard:
    def test_import_sets_numba_cache_dir(self):
        """仅仅 `import audio_separation` 就必须设好 NUMBA_CACHE_DIR。"""
        proc = _run(
            "import os, audio_separation\n"
            "print(os.environ.get('NUMBA_CACHE_DIR', ''))\n"
        )
        assert proc.returncode == 0, proc.stderr
        path = proc.stdout.strip()
        assert path, "import audio_separation 没有设置 NUMBA_CACHE_DIR"
        assert Path(path).is_dir(), f"指向的目录不存在：{path}"

    def test_numba_cache_dir_is_writable(self):
        """必须真的能写 —— 指到一个存在但只读的目录等于没做。"""
        proc = _run(
            "import os, tempfile, audio_separation\n"
            "d = os.environ['NUMBA_CACHE_DIR']\n"
            "p = os.path.join(d, 'probe.tmp')\n"
            "open(p, 'w').close()\n"
            "os.unlink(p)\n"
            "print('WRITABLE')\n"
        )
        assert proc.returncode == 0, proc.stderr
        assert "WRITABLE" in proc.stdout

    def test_existing_value_is_respected(self):
        """外部已经设过就必须尊重 —— 探测子进程会设自己的，不能被覆盖。"""
        script = (
            "import os, tempfile\n"
            "d = tempfile.mkdtemp(prefix='preset_')\n"
            "os.environ['NUMBA_CACHE_DIR'] = d\n"
            "import audio_separation\n"
            "print(os.environ['NUMBA_CACHE_DIR'])\n"
        )
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
            cwd=str(SRC.parent), env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert Path(proc.stdout.strip()).name.startswith("preset_"), (
            f"覆盖了外部已设置的 NUMBA_CACHE_DIR：{proc.stdout!r}"
        )

    @pytest.mark.slow
    def test_importing_bs_roformer_does_not_hang(self):
        """★真正的回归护栏：在没有 NUMBA_CACHE_DIR 的环境里 import bs_roformer。

        这是那个 50 分钟挂死的最小复现。给它一个有限超时 —— 死循环不会抛异常，
        只会耗尽超时，这正是它当初难查的原因。
        """
        proc = _run(
            "import audio_separation  # 施加规避\n"
            "import bs_roformer\n"
            "print('IMPORTED', bs_roformer.__version__ if hasattr(bs_roformer,'__version__') else '')\n",
            timeout=180,
        )
        assert proc.returncode == 0, (
            f"import bs_roformer 未能完成（rc={proc.returncode}），"
            f"很可能退回 numba 静默死循环：\n{proc.stdout}\n{proc.stderr}"
        )
        assert "IMPORTED" in proc.stdout


class TestStreamGuard:
    def test_stdout_is_utf8_after_import(self):
        """入口之后 stdout 必须是 UTF-8，否则日志里的 · ★ 『』 会炸。"""
        proc = _run(
            "import sys, audio_separation\n"
            "print(sys.stdout.encoding)\n"
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().lower().replace("-", "") == "utf8"

    def test_non_ascii_survives_a_pipe(self):
        """日志里固定出现的非 ASCII 字符必须能穿过管道（CLI 的常态）。"""
        proc = _run(
            "import audio_separation\n"
            "print('· 解码 → 『』 ★ ⇒')\n"
        )
        assert proc.returncode == 0, proc.stderr
        assert "· 解码 → 『』 ★ ⇒" in proc.stdout
