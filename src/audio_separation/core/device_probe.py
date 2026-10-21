# -*- coding: utf-8 -*-
"""设备探测子进程（PLAN §2.6 的 L2 层 / §2.8 启动期规避）。

## 为什么单独成文件

这个探测必须在**独立进程**里跑：torch + bs_roformer 的 import 在某些环境下
会挂死（实测过 50 分钟以上的静默死循环），而 GUI 进程绝不能承担这个风险。

原先探测脚本是 `core/env_check.py` 里的一段**字符串**，用
``python -c "<脚本>"`` 跑。那种写法在冻结态下彻底失效 ——
冻结态没有 ``python -c`` 这种跑法，exe 会把 ``-c`` 当成不认识的参数，
于是走进 GUI 分支再也不返回，表现为「探测超时（>90s）」。

现在它是本模块，有两种运行方式：

- 开发态：``python src/audio_separation/core/device_probe.py``
- 冻结态：``audio_separation --probe``（见 ``ui/app.py`` 的入口分派）

## ★ 本模块必须完全自包含

不 import 项目内任何模块。原因：开发态它被当作独立脚本运行
（``python device_probe.py``），此时 ``__package__`` 为空，
相对 import 会直接抛 ImportError；冻结态它被 exe 以 ``--probe`` 调起，
import 路径又完全不同。两种方式共用一份代码，就只能依赖标准库。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings

#: 结果写入路径，由调用方通过环境变量指定（避免 torch 污染 stdout）
RESULT_ENV_VAR = "PROBE_RESULT_FILE"


def _apply_startup_workarounds() -> None:
    """PLAN §2.8 四条规避的前三条，**顺序敏感**。

    1. NUMBA_CACHE_DIR 指到可写临时目录
       —— librosa 的 numba 装饰器在模块执行期就要落缓存，默认位置是
       site-packages/__pycache__；该目录不可写时 numba 会反复尝试建临时文件，
       **不抛异常、不返回**，表现为进程静默挂死。
    2. stdout/stderr 重配为 UTF-8
       —— Windows 默认 GBK，打印 ⇒(U+21D2) 直接崩。
    3. 再 import torch/librosa/bs_roformer
       —— 顺序颠倒则 1、2 失效。
    """
    os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba_cache_")
    for stream in (sys.stdout, sys.stderr):
        # ★ 用 getattr 取，而不是直接 stream.reconfigure(...)：
        #   类型标注里 TextIO 没有声明 reconfigure，直连会触发 Pyright 误报。
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    warnings.filterwarnings("ignore")


def _probe() -> dict:
    """跑一遍探测，返回结果字典。"""
    result = {
        "ok": False,
        "backend": "unknown",
        "device_name": "",
        "torch_version": "",
        "flash_attn": False,
        "error": "",
    }

    try:
        import torch

        result["torch_version"] = torch.__version__

        # ★PLAN §2.14 GPU-only 硬约束：探测不到 GPU 就是终止条件，没有 CPU 兜底。
        # 后端只能从 "+rocm" 本地标签判断 —— ROCm build 的
        # torch.cuda.is_available() 同样返回 True，不能靠它反推。
        if "rocm" in torch.__version__:
            result["backend"] = "rocm"
            result["flash_attn"] = False  # ROCm 不启用 flash_attn
        else:
            result["backend"] = "cuda"
            result["flash_attn"] = True  # CUDA 默认尝试启用

        if not torch.cuda.is_available():
            build = "ROCm" if result["backend"] == "rocm" else "CUDA"
            reason = (
                "torch 报告没有可用的 GPU（当前 torch 是 {ver}，{build} 构建）。"
                "本工具不提供 CPU 模式，GPU 不可用时立即终止。"
            ).format(ver=result["torch_version"], build=build)
            if build == "ROCm":
                reason += (
                    " 请检查显卡驱动与 ROCm 支持列表："
                    "https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html"
                )
            else:
                reason += " 请确认这台机器确实有受支持的独立显卡，以及驱动版本与 torch 构建匹配。"
            result["error"] = reason
            return result

        result["device_name"] = torch.cuda.get_device_name(0)

        from bs_roformer import BSRoformer

        # 一次极小的假推理，捕捉 kernel 层错误（L1 查不到的）。
        # ★必须搬到 GPU 上跑：这一步的全部价值就是验证"GPU 那条路能不能走通"，
        # 留在 CPU 上跑等于什么都没测（ROCm 的 kernel 签名/显存问题一个都抓不到）。
        # 用默认 STFT 设置（n_fft=2048 → 1025 频点），与 DEFAULT_FREQS_PER_BANDS 匹配。
        try:
            model = BSRoformer(
                dim=128,
                depth=1,
                stereo=True,
                num_stems=1,
                time_transformer_depth=1,
                freq_transformer_depth=1,
                flash_attn=result["flash_attn"],
            )
            model.eval().to("cuda")
            x = torch.randn(1, 2, 4096, device="cuda")
            with torch.no_grad():
                y = model(x)
            # ★真搬一次：探测"显存够不够"这件事在 BSRoformer 之后才问才准
            y = y.float().cpu()
            result["ok"] = True
        except Exception as exc:
            # 推理失败：可能是 flash_attn 问题，降级后重试
            if result["flash_attn"]:
                result["flash_attn"] = False
                try:
                    model = BSRoformer(
                        dim=128,
                        depth=1,
                        stereo=True,
                        num_stems=1,
                        time_transformer_depth=1,
                        freq_transformer_depth=1,
                        flash_attn=False,
                    )
                    model.eval().to("cuda")
                    x = torch.randn(1, 2, 4096, device="cuda")
                    with torch.no_grad():
                        y = model(x)
                    y = y.float().cpu()
                    result["ok"] = True
                except Exception as exc2:
                    result["error"] = f"GPU 推理失败（flash_attn=False）：{str(exc2)[:200]}"
            else:
                result["error"] = f"GPU 推理失败：{str(exc)[:200]}"
    except Exception as exc:
        result["error"] = str(exc)[:500]
        import traceback

        result["traceback"] = traceback.format_exc()

    return result


def _write_diagnostics() -> None:
    """把 sys.path 与关键模块的导入结果落到文件（仅当设置了 PROBE_DIAG_FILE）。"""
    target = os.environ.get("PROBE_DIAG_FILE", "")
    if not target:
        return
    import traceback

    lines = ["sys.path:"]
    lines.extend(f"  {p}" for p in sys.path)
    for mod in ("bisect", "mpmath", "sympy", "numpy", "torch", "numba", "librosa",
                "soundfile", "soxr", "pooch", "joblib", "decorator", "lazy_loader"):
        try:
            module = __import__(mod)
            path = getattr(module, "__file__", "")
            lines.append(f"{mod}: OK {path}")
        except Exception:
            lines.append(f"{mod}: FAIL")
            lines.append(traceback.format_exc())
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    """探测入口。结果写入 ``PROBE_RESULT_FILE`` 指向的文件。"""
    _apply_startup_workarounds()
    _write_diagnostics()
    result = _probe()

    target = os.environ.get("PROBE_DIAG_FILE", "")
    if target:
        import traceback

        with open(target, "a", encoding="utf-8") as f:
            f.write("\n\n===== _probe() 结果 =====\n")
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
            f.write("\n")
            if not result.get("ok"):
                f.write("\n===== 完整 traceback =====\n")
                f.write(traceback.format_exc())

    result_file = os.environ.get(RESULT_ENV_VAR, "")
    if result_file:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f)
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
