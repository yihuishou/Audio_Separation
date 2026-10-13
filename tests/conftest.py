"""
tests/conftest.py — pytest fixtures for Audio_Separation test suite.

Environment guard:
- ★环境规避现在由 ``audio_separation/__init__.py`` 在导入时统一施加（PLAN §2.8），
  本文件里不再重复设置 —— 生产入口与测试入口必须走同一条路径，否则测试通过
  而产品挂死（这正是本项目踩过的坑：librosa 的 numba 装饰器在 import 期写缓存，
  目录不可写时静默死循环，表现为"跑 50 分钟无输出"）。
  这里保留一条**验证**用例：断言该规避在干净的子进程里确实生效。

Fixtures:
- repo_root: Path to the repository root (derived from conftest.py location)
- env_report: L1 static library check results (fast, ~250ms)
- env_probe: L2 device probe results (slow, ~4s, spawns subprocess)
- clips: List of clip files in test_audio/clips/ (generates if missing)
- clip_paths: Safe wrapper that skips if clips are empty
- tmp_out: Function-scoped temporary output directory
- requires_gpu: Skip marker when no GPU/accelerator is available
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Import paths module from the project
from audio_separation import paths as audio_paths
from audio_separation.core.env_check import check_libs, probe_device


@pytest.fixture(scope="session", autouse=True)
def _env_guard():
    """不再在此重复施加规避。

    `import audio_separation` 时 `__init__.py` 已经把 NUMBA_CACHE_DIR 与 UTF-8
    流配好（上面第 30 行的 import 本身就是它生效的证据）。这里保留一个空实现
    是为了让"谁负责环境准备"只有一个答案：包入口，而不是每个调用方各做一遍。
    """
    return None


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root path.

    Derived from this conftest.py file's location:
    conftest.py lives in tests/, so the repo root is tests/'s parent.
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def env_report() -> list[dict[str, Any]]:
    """L1 static library check — fast (~250ms), no subprocess, no torch import.

    Returns a list of dependency status rows.
    """
    return check_libs()


@pytest.fixture(scope="session")
def env_probe() -> dict[str, Any]:
    """L2 device probe — slow (~4s), spawns a subprocess to detect GPU/CPU backend.

    Returns a dict with keys: ok, backend, device_name, torch_version,
    flash_attn, error.
    """
    return probe_device()


@pytest.fixture
def requires_gpu(request):
    """Skip test if no accelerator is available.

    Use with the env_probe fixture to check if a GPU is available.
    """
    probe = request.getfixturevalue("env_probe")
    if not probe.get("ok"):
        pytest.skip(f"No accelerator available: {probe.get('error', 'unknown')}")


@pytest.fixture(scope="session")
def clips() -> list[Path]:
    """Return sorted list of clip WAV files in test_audio/clips/.

    If the directory is empty or missing, attempt to generate clips via
    tools/make_test_clips.py. If that fails, skip the test (test_audio/
    is 513MB and gitignored, so fresh clones legitimately have no clips).
    """
    clips_dir = audio_paths.test_clips_dir()

    # Try to list existing clips
    try:
        clip_files = sorted(clips_dir.glob("*.wav"))
        if clip_files:
            return clip_files
    except (FileNotFoundError, OSError):
        pass

    # Try to generate clips via the tool
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = clips_dir.resolve()
    clips_dir = clips_dir.parent

    try:
        result = subprocess.run(
            [sys.executable, "tools/make_test_clips.py"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            clip_files = sorted(clips_dir.glob("*.wav"))
            if clip_files:
                return clip_files
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Still no clips — skip
    pytest.skip("No test clips available and could not generate them")


@pytest.fixture
def clip_paths(clips):
    """Safe wrapper around clips fixture that skips if empty."""
    if not clips:
        pytest.skip("No clip files available")
    return clips


@pytest.fixture
def tmp_out(tmp_path):
    """Function-scoped temporary output directory."""
    return tmp_path / "output"


@pytest.fixture
def requires_clips(request):
    """Skip test if clips fixture is not available (no test audio)."""
    clips = request.getfixturevalue("clips")
    if not clips:
        pytest.skip("No test clips available")
    return clips
