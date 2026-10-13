"""PLAN §2.14 GPU-only 硬约束的契约测试。

为什么单独一个文件：这条约束横跨四个模块（env_check / pipeline / model / cli），
分散在各文件的测试里没人能一眼看全；更要紧的是，它是**否定式**契约 ——
"不许有 CPU 兜底"没法靠正向断言表达，只能靠"如果它回来，测试就红"来守住。
省略任何一条，下一个人都会觉得"加个回退总比直接崩好吧"。

全部用例都在子进程里跑或只做静态检查，因为：
  1. `build_model` 的设备校验发生在 `import torch` 之前，但同进程先 import 过
     一次 torch 再验证"不会 import"，这条断言本身就没意义了；
  2. 本文件不得拖慢快速测试集 —— 它属于 `-m 'not slow'`。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parent.parent / "src" / "audio_separation" / "core"


def _source(name: str) -> str:
    return (CORE / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name))


def _string_literals(tree: ast.AST) -> set[str]:
    """收集所有字符串常量 —— 含 docstring。

    静态检查"源码里没写过某个词"时必须先剥掉字符串，否则模块 docstring 里
    一句"我们不做 CPU 回退"就会把自己的断言判成违规（这个坑本文件早先踩过）。
    """
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


# ---------------------------------------------------------------------------
# 设备入口：任何显式传 cpu 的路径都必须当场拒绝
# ---------------------------------------------------------------------------

class TestDeviceArgumentsAreRejected:
    def test_build_model_rejects_cpu(self):
        """build_model(device='cpu') 必须在 import bs_roformer 之前就抛。

        用子进程跑：既保证"校验发生在 import torch 之前"这条顺序约束真的成立，
        也顺带证明导入本模块不需要 torch。
        """
        script = (
            "import sys\n"
            "from audio_separation.core.model import build_model\n"
            "assert 'torch' not in sys.modules, 'import 本模块竟然拖进了 torch'\n"
            "try:\n"
            "    build_model({}, device='cpu')\n"
            "except ValueError as exc:\n"
            "    print('REJECTED:', exc)\n"
            "    raise SystemExit(0)\n"
            "except BaseException as exc:\n"
            "    print('WRONG-EXC:', type(exc).__name__, exc)\n"
            "    raise SystemExit(2)\n"
            "print('NOT-RAISED')\n"
            "raise SystemExit(3)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert proc.returncode == 0, (
            f"build_model 未按预期拒绝 cpu：rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )
        assert "GPU" in proc.stdout, f"拒绝理由没有点明 GPU-only：{proc.stdout!r}"

    @pytest.mark.parametrize("device", ["cpu", "mps", "xpu", ""])
    def test_resolve_device_rejects_non_cuda(self, device):
        """_resolve_device 除了 auto / cuda 之外一律终止，不得静默接受。"""
        from audio_separation.core.pipeline import PipelineError, _resolve_device

        with pytest.raises(PipelineError) as info:
            _resolve_device(device, log=lambda *_a: None)
        assert "GPU" in str(info.value)

    def test_resolve_device_terminates_when_gpu_absent(self, monkeypatch):
        """探测失败 ⇒ PipelineError（终止），而不是回退到某个能跑的后端。"""
        import audio_separation.core.env_check as env_check
        from audio_separation.core import pipeline

        monkeypatch.setattr(
            env_check,
            "probe_device",
            lambda *a, **k: {
                "ok": False, "backend": "unknown", "device_name": "",
                "torch_version": "", "flash_attn": False,
                "error": "探测超时（>90s）",
            },
        )
        with pytest.raises(pipeline.PipelineError) as info:
            pipeline._resolve_device("auto", log=lambda *_a: None)
        message = str(info.value)
        assert "已终止" in message
        # 失败原因必须原样透出，不能被换成一句笼统的"设备不可用"
        assert "探测超时" in message

    def test_resolve_device_rejects_untrustworthy_backend(self, monkeypatch):
        """探测自称成功却给出 rocm/cuda 之外的后端 —— 当成探测不可信，不猜。"""
        import audio_separation.core.env_check as env_check
        from audio_separation.core import pipeline

        monkeypatch.setattr(
            env_check, "probe_device",
            lambda *a, **k: {
                "ok": True, "backend": "cpu", "device_name": "CPU",
                "torch_version": "2.9.1", "flash_attn": False, "error": "",
            },
        )
        with pytest.raises(pipeline.PipelineError) as info:
            pipeline._resolve_device("auto", log=lambda *_a: None)
        assert "不可信" in str(info.value)


# ---------------------------------------------------------------------------
# CLI 层：argparse 层面就挡住，用户不必等到运行时才知道 cpu 不支持
# ---------------------------------------------------------------------------

class TestCliSurface:
    def test_cli_rejects_device_cpu(self):
        """--device cpu 必须在 argparse 阶段退出（退出码 2）。

        注意 argparse 的 error 行只会打印 `invalid choice`，**不会**带上
        `--device` 的 help 文本 —— 所以只能断言 usage 里那组 choices，
        提示文案要靠 `cli.py --help` 单独验。
        """
        proc = subprocess.run(
            [sys.executable, "-m", "audio_separation.core.cli",
             "--device", "cpu", "-i", "x.wav"],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert proc.returncode == 2, (
            f"--device cpu 应被 argparse 拒绝，实际 rc={proc.returncode}\n{proc.stdout}"
        )
        combined = proc.stdout + proc.stderr
        assert "invalid choice: 'cpu'" in combined
        assert "--device {auto,cuda}" in combined, (
            f"usage 里的可选值应只有 auto/cuda：{combined!r}"
        )

    def test_cli_help_states_the_gpu_only_rule(self):
        """帮助文本必须自己讲清"没有 cpu"，否则用户看到 choices 只能靠猜。"""
        proc = subprocess.run(
            [sys.executable, "-m", "audio_separation.core.cli", "--help"],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "2.14" in proc.stdout or "GPU" in proc.stdout, (
            f"--help 里没有说明 GPU-only：{proc.stdout!r}"
        )

    def test_model_main_rejects_device_cpu(self):
        proc = subprocess.run(
            [sys.executable, "-m", "audio_separation.core.model",
             "--check", "v2", "--device", "cpu"],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert proc.returncode == 2, (
            f"--device cpu 应被 argparse 拒绝，实际 rc={proc.returncode}\n{proc.stdout}"
        )


# ---------------------------------------------------------------------------
# 源码层：不存在被注释掉、但还活着的 CPU 回退分支
# ---------------------------------------------------------------------------

class TestNoCpuFallbackInSource:
    @pytest.mark.parametrize("name", ["pipeline.py", "env_check.py"])
    def test_no_cpu_return_statement(self, name):
        """任何 `return "cpu"` 都是回退分支，必须已经消失。"""
        tree = _tree(name)
        offenders = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and node.value.value == "cpu"
        ]
        assert not offenders, f"{name} 仍有返回 CPU 的分支：{offenders}"

    def test_probe_script_has_no_cpu_backend(self):
        """探测子进程的脚本源码（一个字符串常量）里，backend 只能被赋成 rocm / cuda。"""
        from audio_separation.core.env_check import _build_probe_script

        tree = ast.parse(_build_probe_script())
        assigned = {
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Name)
                and t.value.id == "result"
                and isinstance(t.slice, ast.Constant)
                and t.slice.value == "backend"
                for t in node.targets
            )
        }
        assert assigned <= {"rocm", "cuda", "unknown"}, (
            f"探测脚本把 result['backend'] 赋成了意外值：{assigned}"
        )

    def test_probe_script_skips_inference_without_gpu(self):
        """没有 GPU 时探测脚本必须止步于 error，不能继续走到 bs_roformer 假推理。

        判据是结构性的：GPU 分支里的 `model.eval().to("cuda")` 必须嵌套在
        `if torch.cuda.is_available():`（或其 else 分支）之下。
        """
        from audio_separation.core.env_check import _build_probe_script

        tree = ast.parse(_build_probe_script())

        def _is_is_available(node: ast.AST) -> bool:
            """识别 `torch.cuda.is_available()`，容许外层裹一层 `not`。"""
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                node = node.operand
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "is_available"
            )

        guards = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and _is_is_available(node.test)
        ]
        assert guards, "探测脚本不再检查 torch.cuda.is_available()，GPU 缺失会一路走下去"

        cuda_moves = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
            and any(isinstance(a, ast.Constant) and a.value == "cuda" for a in node.args)
        ]
        assert cuda_moves, "探测脚本的假推理没有搬到 GPU 上，那这一步等于没测"

        # 守卫是 cuda 调用的**祖先**，所以要向上走而不是向下走
        parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node

        def _ancestors(node: ast.AST) -> list[ast.AST]:
            chain: list[ast.AST] = []
            while (owner := parent.get(node)) is not None:
                chain.append(owner)
                node = owner
            return chain

        assert any(guard in _ancestors(cuda_node) for cuda_node in cuda_moves for guard in guards), (
            "GPU 假推理没有受 is_available() 保护 —— 没有 GPU 时它也会被执行"
        )

    @pytest.mark.parametrize(
        "name",
        ["pipeline.py", "model.py", "separator.py", "watchdog.py", "cli.py", "env_check.py"],
    )
    def test_cpu_literals_only_appear_in_allowed_contexts(self, name):
        """每一处字符串 "cpu" 都必须落在允许的上下文里，否则就是漏网的后端逻辑。

        允许的三类：
          - ``torch.load(..., map_location="cpu")`` —— 加载终点，下一行就 ``.to("cuda")``；
          - ``.to("cpu")`` / ``.cpu()`` —— 把张量搬回主机做加窗累加；
          - docstring 与注释（AST 里是字符串常量，但靠"所在调用点"就能区分开）。
        """
        tree = _tree(name)
        parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node

        offenders: list[str] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and node.value == "cpu"):
                continue
            owner = parent.get(node)
            if isinstance(owner, ast.keyword) and owner.arg == "map_location":
                continue
            if (
                isinstance(owner, ast.Call)
                and isinstance(owner.func, ast.Attribute)
                and owner.func.attr in {"to", "cpu", "device", "map_location"}
            ):
                continue
            offenders.append(ast.unparse(owner) if owner is not None else "<module>")

        assert not offenders, (
            f"{name} 里的 'cpu' 出现在允许范围之外：{offenders}"
        )

    def test_model_does_not_import_torch_at_module_level(self):
        """§2.14 之后 torch 是硬依赖，但本模块仍应能被只读元数据地 import。

        延迟导入让 torch 缺失/损坏时 `list_models()` 与诊断信息照样可用，
        也让快速测试集不必为断言一个默认值拖进 1.5 s 的 torch 导入。
        """
        tree = _tree("model.py")
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert not any(a.name == "torch" for a in node.names), (
                    "model.py 顶层又 import torch 了，应保持延迟导入"
                )

    def test_device_defaults_never_point_at_cpu(self):
        """默认值本身就是契约的一部分：写错默认等于悄悄给了一条 CPU 路径。"""
        import inspect

        from audio_separation.core import model, pipeline, separator

        for func in (separator.separate_vocals, model.build_model, pipeline.separate_file):
            default = inspect.signature(func).parameters["device"].default
            assert default in ("cuda", "auto"), (
                f"{func.__name__} 的 device 默认值是 {default!r}，"
                f"应为 'cuda' 或 'auto'（PLAN §2.14）"
            )
