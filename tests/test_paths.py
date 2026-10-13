"""
tests/test_paths.py — Path source-of-truth contract tests.

Tests the paths module's core API without importing torch or any
accelerator-dependent modules. All tests are fast (<1s).
"""

import os
import sys

import pytest

from audio_separation import paths as audio_paths
from audio_separation.paths import (
    ENV_OUTPUT,
    ENV_ROOT,
    ModelInfo,
    ROOT_MARKERS,
    ResourceNotFound,
    ffmpeg_bin,
    list_models,
    output_dir,
    repo_root,
)

# ★audio_paths.test_audio_dir / test_clips_dir 不能裸 import 到本模块。
# 它们的名字以 test_ 开头，一进本模块命名空间 pytest 就会把它们当测试收集，
# 直接调用后返回 Path 而不抛断言——只会刷出 PytestReturnNotNoneWarning。
# 统一走 audio_paths. 前缀访问。


class TestPathConstants:
    """Verify the root marker and environment variable constants."""

    def test_root_markers(self):
        """ROOT_MARKERS must be ('models', 'ffmpeg') — the two required
        top-level directories that identify the repo root."""
        assert ROOT_MARKERS == ("models", "ffmpeg")

    def test_env_root(self):
        """ENV_ROOT must be 'AUDIO_SEPARATION_ROOT'."""
        assert ENV_ROOT == "AUDIO_SEPARATION_ROOT"

    def test_env_output(self):
        """ENV_OUTPUT must be 'AUDIO_SEPARATION_OUTPUT'."""
        assert ENV_OUTPUT == "AUDIO_SEPARATION_OUTPUT"


class TestFFmpegPaths:
    """Test ffmpeg binary path resolution."""

    def test_ffmpeg_bin_filename_matches_platform(self):
        """ffmpeg_bin() must return a path ending with the platform-specific
        executable name: 'ffmpeg.exe' on Windows, 'ffmpeg' on Linux/macOS."""
        path = ffmpeg_bin()
        if sys.platform == "win32":
            assert path.name == "ffmpeg.exe", f"Expected ffmpeg.exe on Windows, got {path.name}"
        else:
            assert path.name == "ffmpeg", f"Expected ffmpeg on {sys.platform}, got {path.name}"

    def test_ffprobe_bin_filename_matches_platform(self):
        """ffprobe_bin() must return a path ending with the platform-specific
        executable name."""
        path = audio_paths.ffprobe_bin()
        if sys.platform == "win32":
            assert path.name == "ffprobe.exe", f"Expected ffprobe.exe on Windows, got {path.name}"
        else:
            assert path.name == "ffprobe", f"Expected ffprobe on {sys.platform}, got {path.name}"


class TestRepoRoot:
    """Test repo root detection."""

    def test_repo_root_returns_existing_directory(self):
        """repo_root() must return an existing directory."""
        root = repo_root()
        assert root.is_dir(), f"repo_root() returned non-directory: {root}"

    def test_repo_root_contains_root_markers(self):
        """repo_root() must return a directory containing all ROOT_MARKERS."""
        root = repo_root()
        for marker in ROOT_MARKERS:
            marker_path = root / marker
            assert marker_path.exists(), f"repo_root() {root} does not contain {marker}/"

    def test_env_root_override_with_invalid_path_raises(self):
        """Setting AUDIO_SEPARATION_ROOT to a directory without root markers
        must raise ResourceNotFound."""
        import tempfile

        # Create a temp directory without the required markers
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AUDIO_SEPARATION_ROOT"] = tmpdir
            try:
                # Clear the cache to force re-resolution
                audio_paths.reset_cache()
                with pytest.raises(ResourceNotFound):
                    repo_root()
            finally:
                # Clean up
                del os.environ["AUDIO_SEPARATION_ROOT"]
                audio_paths.reset_cache()


class TestPathHelpers:
    """Test path helper functions."""

    def test_output_dir_anchored_under_repo_root(self):
        """output_dir() must be anchored under repo_root(), not use absolute
        paths or '..' traversal."""
        root = repo_root()
        out = output_dir()
        # Resolve both to absolute paths for comparison
        assert out.resolve().is_relative_to(root.resolve()), (
            f"output_dir() {out} is not under repo_root() {root}"
        )

    def test_test_audio_dir_anchored_under_repo_root(self):
        """test_audio_dir() must be under repo_root()."""
        root = repo_root()
        audio_dir = audio_paths.test_audio_dir()
        # test_audio/ might not exist yet, so check the path relationship
        assert audio_dir.resolve().is_relative_to(root.resolve()), (
            f"test_audio_dir() {audio_dir} is not under repo_root() {root}"
        )

    def test_test_clips_dir_anchored_under_repo_root(self):
        """test_clips_dir() must be under repo_root()."""
        root = repo_root()
        clips_dir = audio_paths.test_clips_dir()
        assert clips_dir.resolve().is_relative_to(root.resolve()), (
            f"test_clips_dir() {clips_dir} is not under repo_root() {root}"
        )


class TestModelDiscovery:
    """Test model directory discovery."""

    def test_list_models_returns_only_complete_models(self):
        """list_models() must return only models that have BOTH a .yaml and
        a same-stem .ckpt file."""
        models = list_models()

        for model in models:
            assert isinstance(model, ModelInfo), f"Expected ModelInfo, got {type(model)}"
            assert model.yaml_path.exists(), f"yaml missing for {model.key}"
            assert model.ckpt_path.exists(), f"ckpt missing for {model.key}"

    def test_list_models_includes_v2_and_v3_when_present(self):
        """If v2 and v3 model files exist, list_models() must include them."""
        models = list_models()
        model_keys = [m.key for m in models]

        # Check if v2 and v3 files actually exist on disk
        root = repo_root()
        models_dir = root / "models" / "bs_roformer_revive"

        v2_ckpt = models_dir / "bs_roformer_revive_v2_by_unwa.ckpt"
        v3_ckpt = models_dir / "bs_roformer_revive_v3_by_unwa.ckpt"

        if v2_ckpt.exists():
            assert "v2" in model_keys, "v2 .ckpt exists but not in list_models()"

        if v3_ckpt.exists():
            assert "v3" in model_keys, "v3 .ckpt exists but not in list_models()"
