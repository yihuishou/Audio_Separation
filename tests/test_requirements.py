# -*- coding: utf-8 -*-
"""依赖清单的 schema 与 R21/R22 契约测试。

R21（§2.13 G 组）：torch 只能出现在 ``REQUIREMENTS`` 里、``group == "G"``、
``is_detect_only`` 为真；并且**不得**出现在 pyproject.toml 与 requirements.txt
的依赖行中（只允许出现在注释里）。

R22：检测层不得 import 目标模块（``importlib.import_module``）、不得调 pip、
不得用 ``os.system``。这三者任一出现，§2.6 的「只检测不安装」承诺就作废了。

★注意本文件所有读源码/读文本的地方都必须显式指定 ``encoding="utf-8"``：
Windows 的 ``locale`` 编码是 GBK，而项目源码与文档里大量使用中文，
不指定就会 UnicodeDecodeError——本文件的第一版就是这么炸的。
"""

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

from audio_separation.core import requirements as req_module

# ★判断「这一行是不是 torch 这个发行包」的正规做法。
# 之前用 startswith("torch") 把 torch-einops-utils 一起误伤了。
_TORCH_SPEC_RE = re.compile(r"^torch\s*(?:$|[<>=!~;\[@ ])")


def is_torch_spec(text: str) -> bool:
    """判断一行依赖声明是否指的就是 torch 本身（排除 torch-einops-utils 等同前缀包）。"""
    return bool(_TORCH_SPEC_RE.match(text.strip().lower()))


def req_source_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "audio_separation" / "core" / name


def read_text_utf8(path: Path) -> str:
    """统一以 UTF-8 读取，避免 Windows 上 GBK locale 解码失败。"""
    return path.read_text(encoding="utf-8")


class TestRequirementsSchema:
    """REQUIREMENTS 列表的结构约束。"""

    def test_all_entries_have_required_keys(self):
        """除 F 组（内嵌资源无 pip 版本号）外，每条都必须齐五个必备键。"""
        allowed_groups = {"A", "B", "C", "D", "E", "F", "G"}
        for req in req_module.REQUIREMENTS:
            name = req.get("name", "<无名>")
            assert req.get("group") in allowed_groups, f"{name} 组别非法：{req.get('group')}"
            assert req.get("level") in {"FAIL", "WARN", "INFO"}, f"{name} 级别非法：{req.get('level')}"
            # ★install_hint 必须是非空字符串，且永远只是给人看的提示，不得是别的东西
            hint = req.get("install_hint")
            assert isinstance(hint, str) and hint.strip(), f"{name} 缺少 install_hint 字符串"
            if req["group"] != "F":
                assert req.get("min_version"), f"{name} 缺少 min_version"

    def test_entry_count(self):
        """条目数固定，防止手工增删时静默漏项。"""
        assert len(req_module.REQUIREMENTS) == 22

    def test_group_coverage(self):
        """A-G 七组都必须在清单里出现过。"""
        groups = {req["group"] for req in req_module.REQUIREMENTS}
        assert groups == {"A", "B", "C", "D", "E", "F", "G"}

    def test_no_duplicate_names(self):
        names = [req["name"].lower() for req in req_module.REQUIREMENTS]
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"清单里有重复条目：{dupes}"

    def test_torch_min_version_is_capability_floor_not_local_build(self):
        """★torch 的 min_version 不能抄本机版本串。

        本机是 ``2.9.1+rocmsdk20260116``；把它当成硬性下限，
        Linux 上装 ``2.7.1+cu124`` 就会被误判成「版本不够」。
        """
        torch_req = next(r for r in req_module.REQUIREMENTS if r["name"].lower() == "torch")
        floor = torch_req["min_version"]
        assert "+" not in floor, f"torch 的 min_version 带了本地构建标签：{floor}"
        assert floor == "2.0", f"torch 能力下限应取 BS-RoFormer 声明的 2.0，实为 {floor}"


class TestR21TorchIsDetectOnly:
    """R21：torch 只检测、不安装。"""

    def test_torch_in_group_g_with_detect_only_flag(self):
        """torch 必须在清单里，且明确标成 G 组 + 只检测。"""
        torch_req = next(r for r in req_module.REQUIREMENTS if r["name"].lower() == "torch")
        assert torch_req["group"] == "G", "torch 必须归入 G 组（只检测不安装）"
        assert torch_req.get("is_detect_only") is True, "torch 必须标记 is_detect_only"
        assert torch_req["level"] == "FAIL", "torch 缺失属致命问题，应为 FAIL"

    def test_torch_install_hints_cover_gpu_backends(self):
        """torch 装哪个 GPU 后端由硬件决定，两条命令必须都在，供 L3 面板按探测结果选一行。"""
        torch_req = next(r for r in req_module.REQUIREMENTS if r["name"].lower() == "torch")
        hints = torch_req.get("install_hints")
        assert isinstance(hints, dict), "torch 必须带 install_hints 字典"
        assert set(hints) == {"rocm", "cuda"}
        for backend, cmd in hints.items():
            assert "download.pytorch.org" in cmd, f"{backend} 的安装命令指向了非官方源：{cmd}"
            assert req_module.TORCH_INSTALL_HINTS[backend] == cmd

    def test_torch_has_no_cpu_install_hint(self):
        """PLAN §2.14：不得提供 CPU-only wheel 的安装命令。

        那条命令是一条死路 —— 装出来的 torch 能过 L1（包在、版本对）却过不了 L2，
        用户只会得到一个更晚、更难懂的失败。
        """
        torch_req = next(r for r in req_module.REQUIREMENTS if r["name"].lower() == "torch")
        hints = torch_req.get("install_hints") or {}
        assert "cpu" not in hints, "PLAN §2.14 是 GPU-only，不得再给 CPU wheel 的安装命令"
        assert "cpu" not in req_module.TORCH_INSTALL_HINTS

    def test_pyproject_mentions_only_gpu_wheels(self):
        """requirements.txt / pyproject.toml 注释里给的 torch 命令也不该出现 cpu 索引源。"""
        for name in ("requirements.txt", "pyproject.toml"):
            path = Path(__file__).resolve().parent.parent / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            assert "whl/cpu" not in text, (
                f"{name} 里出现了 CPU-only wheel 的索引源，与 PLAN §2.14 的 GPU-only 冲突"
            )

    def test_torch_not_in_pyproject_dependencies(self):
        """R21：torch 不得进 pyproject.toml 的 dependencies。"""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if not pyproject_path.exists():
            pytest.skip("pyproject.toml 不存在")

        with pyproject_path.open("rb") as fh:
            pyproject = tomllib.load(fh)

        deps = pyproject.get("project", {}).get("dependencies", [])
        offenders = [d for d in deps if is_torch_spec(d)]
        assert not offenders, f"torch 不得由 pip 自动安装，但它出现在 dependencies 里：{offenders}"

    def test_torch_not_in_requirements_txt(self):
        """R21：torch 不得成为 requirements.txt 的依赖行（注释里给命令是可以的）。"""
        req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
        if not req_path.exists():
            pytest.skip("requirements.txt 不存在")

        offenders = []
        for line in read_text_utf8(req_path).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            if is_torch_spec(stripped):
                offenders.append(stripped)

        assert not offenders, f"torch 不得被自动安装，但它出现在 requirements.txt 里：{offenders}"

    def test_other_torch_prefixed_packages_are_still_listed(self):
        """★反向对照：torch-einops-utils 必须仍在依赖里。

        上面的 is_torch_spec 修的就是「用 startswith 把这一条也误伤」的 bug；
        这条测试守住修复本身，防止以后有人图省事又改回 startswith。
        """
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if not pyproject_path.exists():
            pytest.skip("pyproject.toml 不存在")
        with pyproject_path.open("rb") as fh:
            deps = tomllib.load(fh).get("project", {}).get("dependencies", [])
        assert any("torch-einops-utils" in d for d in deps), (
            "torch-einops-utils 是 BS-RoFormer 的传递依赖，必须正常安装；"
            "它被误判成 torch 是 is_torch_spec 退化成 startswith 的信号"
        )


class TestR22DetectionPathIsReadOnly:
    """R22：检测层只读——不 import 目标模块、不碰 pip。

    ★用 ``ast`` 判定而不是正则搜源码。
    早先的正则版要先把注释和字符串剥掉，而「剥离」这件事本身很脆：
    多行 docstring 的续行一旦漏出来，源码里那句说明
    「避免在本进程 import torch」就会被自己判成违规——第一版就是这么假失败的。
    AST 天然只反映真实代码，注释和字符串进不来。
    """

    FILES = ("env_check.py", "requirements.py")
    #: 加载这些模块会在本机触发 ROCm 工具链对含空格路径的 bug 而挂死 >120 s
    HEAVY_MODULES = {"torch", "bs_roformer", "torchaudio"}
    SUBPROCESS_ATTRS = {"run", "call", "Popen", "check_call", "check_output"}

    @classmethod
    def _tree(cls, name: str) -> ast.Module:
        path = req_source_path(name)
        if not path.exists():
            pytest.skip(f"{name} 不存在")
        return ast.parse(read_text_utf8(path), filename=str(path))

    @staticmethod
    def _dotted(node: ast.AST) -> str:
        """属性链还原为点分字符串；不是属性链（调用、字面量等）时返回空串。"""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return ""

    @classmethod
    def _imported_modules(cls, tree: ast.Module) -> set[str]:
        """收集文件里真正被 import 的顶层模块名。"""
        mods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
        return mods

    @classmethod
    def _calls(cls, tree: ast.Module) -> list[tuple[str, ast.Call]]:
        """收集所有「函数调用」，元素是 (点分函数名, 调用节点)。"""
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                found.append((cls._dotted(node.func), node))
        return found

    @staticmethod
    def _string_literals(node: ast.AST) -> list[str]:
        return [
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]

    @pytest.mark.parametrize("name", FILES)
    def test_does_not_import_heavy_modules(self, name):
        """检测层绝不能真的 import torch / bs_roformer。

        L2 的设备探测是另起子进程做的（见 env_check.probe_device），
        本进程必须保持干净，否则会把 UI 进程一起拖挂。
        """
        offenders = self._imported_modules(self._tree(name)) & self.HEAVY_MODULES
        assert not offenders, f"{name} 导入了 {offenders}，检测层会触发本机 ROCm 挂死"

    @pytest.mark.parametrize("name", FILES)
    def test_does_not_use_importlib_import_module(self, name):
        """禁止 ``importlib.import_module()``——「只查元数据」这条线不能破。"""
        for dotted, call in self._calls(self._tree(name)):
            assert dotted != "importlib.import_module", (
                f"{name} 调用了 importlib.import_module()，应改用 importlib.metadata.version()"
            )

    @pytest.mark.parametrize("name", FILES)
    def test_does_not_import_pip(self, name):
        """禁止 ``import pip`` / ``pip._internal`` 这类绕过子进程直接驱动 pip 的路子。"""
        tree = self._tree(name)
        assert "pip" not in self._imported_modules(tree), f"{name} 导入了 pip 模块"
        assert not any(
            dotted.startswith("pip.") for dotted, _ in self._calls(tree)
        ), f"{name} 引用了 pip 的内部 API"

    @pytest.mark.parametrize("name", FILES)
    def test_never_shells_out_to_pip(self, name):
        """★核心断言：无论通过 subprocess 还是 os.system，都不许执行 pip。"""
        tree = self._tree(name)
        for dotted, call in self._calls(tree):
            is_shell_out = dotted == "os.system" or (
                dotted.startswith("subprocess.") and dotted.split(".")[-1] in self.SUBPROCESS_ATTRS
            )
            if not is_shell_out:
                continue
            blob = " ".join(self._string_literals(call)).lower()
            assert "pip" not in blob, (
                f"{name} 通过 {dotted}() 执行了 pip（参数字面量：{blob!r}）；"
                "§2.6 承诺检测层只给可复制命令，不代为安装"
            )

    @pytest.mark.parametrize("name", FILES)
    def test_no_sys_executable_with_pip(self, name):
        """``[sys.executable, "-m", "pip", ...]`` 绕过上面那条断言的常见写法。"""
        for dotted, call in self._calls(self._tree(name)):
            if not dotted.startswith("subprocess."):
                continue
            blob = " ".join(self._string_literals(call)).lower()
            assert not (dotted.endswith(".") or "-m" in blob and "pip" in blob), (
                f"{name} 疑似以 python -m pip 的方式调用了 pip"
            )

    @pytest.mark.parametrize("name", FILES)
    def test_install_hints_are_pure_strings(self, name):
        """★install_hint 只能是给人看的字符串——这是 R22 的数据面保证。

        面板把 remedy 直接显示给用户；若哪天有人塞进可调用对象，
        「点一下修复」就变成了「点一下执行」。
        """
        for entry in req_module.REQUIREMENTS:
            assert isinstance(entry["install_hint"], str), (
                f"{entry['name']} 的 install_hint 不是字符串，类型为 {type(entry['install_hint'])}"
            )
        if name == "requirements.py":
            for backend, hint in req_module.TORCH_INSTALL_HINTS.items():
                assert isinstance(hint, str) and hint.strip(), f"torch {backend} 安装提示为空"


class TestVersionComparison:
    """版本号解析：本地构建标签必须被正确剥离，否则 torch 会被误判。"""

    def test_version_tuple_strips_local_build_labels(self):
        from audio_separation.core.env_check import _version_tuple

        assert _version_tuple("2.9.1+rocmsdk20260116") == (2, 9, 1)
        assert _version_tuple("2.4.3") == (2, 4, 3)
        assert _version_tuple("2.4.3b1") == (2, 4, 3)

    def test_below_detects_old_versions(self):
        from audio_separation.core.env_check import _below

        assert _below("1.26.4", "2.4.3") is True
        assert _below("2.4.3", "2.4.3") is False
        assert _below("2.9.1+rocmsdk20260116", "2.0") is False

    def test_unparseable_version_is_not_reported_as_too_old(self):
        """★解析不出来时宁可放过，也不误报——面板在骗人比宽松更糟。"""
        from audio_separation.core.env_check import _below

        assert _below("unknown", "2.0") is False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
