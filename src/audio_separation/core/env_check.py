# -*- coding: utf-8 -*-
"""环境检测 —— 三层架构（PLAN §2.6）。

L1 · 静态库检测：importlib.metadata.version() 只读安装元数据，不 import 任何目标模块。
    本机实测 `import bs_roformer` 会挂死（ROCm 工具链对 Program Files 空格路径的 bug）。
    若 L1 也走 import，torch 一挂全盘皆黑。
    ★本层只读状态，从不执行 pip install，也从不 import 被检查的包。

L2 · 动态设备探测：子进程 + 超时。依次试 import torch → cuda.is_available() →
    get_device_name → import bs_roformer → 一次极小的假推理（捕捉 kernel 层错误）。
    每一步都带超时（默认 90s），超时按 FAIL 处理。结果缓存（lru_cache + force=True 逃生口）。
    ★子进程入口应用 PLAN §2.8 四条规避，顺序敏感。

L3 · GUI 面板数据结构：build_report_rows() 返回字典列表，供 PySide6 表格渲染。
    diagnostic_text() 拼接完整可复制的诊断报告。
"""

import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache

from . import requirements as req


# ---------------------------------------------------------------------------
# L1 · 静态库检测（不 import 任何目标模块）
# ---------------------------------------------------------------------------

def _version_tuple(version: str) -> tuple[int, ...]:
    """把版本串拆成可比较的整数元组。

    只取前导的数字段，丢掉 `+rocmsdk20260116` / `cu124` / `rc1` 这类本地标签：
    本地标签不代表上游版本序，拿它比较会把 `2.7.1+cu124` 判成小于 `2.9.1`。
    """
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _below(version: str, floor: str) -> bool:
    """实际版本是否低于清单里的下限。版本串解析不出数字时返回 False。

    解析失败宁可放过也不误报：一份非标准版本串（比如源码树直接装的包）不该
    让整个自检面板变红。
    """
    if not floor:
        return False
    got, want = _version_tuple(version), _version_tuple(floor)
    if not got or not want:
        return False
    width = max(len(got), len(want))
    return got + (0,) * (width - len(got)) < want + (0,) * (width - len(want))


def check_libs() -> list[dict]:
    """返回完整的 L1 检测报告，每个条目包含 item/level/detail/remedy。

    ★只使用 importlib.metadata.version(name)，不执行 importlib.import_module
    或 plain import。这是关键：`import bs_roformer` 在本机会挂死（>120s）。
    """
    results = []

    # Python 版本检查（不是 pip 包，单独处理）
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        results.append({
            "item": "Python",
            "level": "PASS",
            "detail": py_version,
            "remedy": "",
        })
    else:
        results.append({
            "item": "Python",
            "level": "FAIL",
            "detail": py_version + "（需要 ≥3.11）",
            "remedy": "安装 Python 3.11 或更新版本",
        })

    # 遍历依赖清单，逐项检查
    for entry in req.REQUIREMENTS:
        if entry.get("is_python"):
            continue  # Python 已单独处理

        # F 组：项目内嵌资源，不是 pip 包，不检查
        if entry.get("is_embedded"):
            results.append({
                "item": entry["name"],
                "level": "INFO",
                "detail": "项目内嵌资源（随仓库分发）",
                "remedy": "",
            })
            continue

        name = entry["name"]
        group = entry["group"]
        min_version = entry.get("min_version", "")
        level = entry.get("level", "INFO")
        install_hint = entry.get("install_hint", "")
        install_hints = entry.get("install_hints", {})

        # ★只读元数据，不 import 目标模块
        try:
            version = importlib.metadata.version(name)
            too_old = _below(version, min_version)
            # torch 特殊处理：额外从版本串识别后端，remedy 给按后端分支的命令字典
            if name == "torch":
                backend = _detect_torch_backend(version)
                item = f"torch（★只检测不安装，{group} 组）"
                if too_old:
                    results.append({
                        "item": item,
                        "level": "FAIL",
                        "detail": f"{version}（{backend}，版本过低，需要 ≥ {min_version}）",
                        "remedy": install_hints,
                    })
                else:
                    results.append({
                        "item": item,
                        "level": "PASS",
                        "detail": f"{version}（{backend}）",
                        "remedy": "",
                    })
            elif too_old:
                results.append({
                    "item": name,
                    "level": level,
                    "detail": f"{version}（版本过低，需要 ≥ {min_version}）",
                    "remedy": install_hint,
                })
            else:
                floor_note = f"（需要 ≥ {min_version}）" if min_version else ""
                results.append({
                    "item": name,
                    "level": "PASS",
                    "detail": f"{version}{floor_note}",
                    "remedy": "",
                })
        except importlib.metadata.PackageNotFoundError:
            # torch 缺失时给出后端分支的安装提示（供 L3 面板按后端选择）
            if name == "torch" and install_hints:
                results.append({
                    "item": "torch（★只检测不安装，G 组）",
                    "level": "FAIL",
                    "detail": "未安装",
                    "remedy": install_hints,  # dict，L3 面板按后端选取
                })
            else:
                results.append({
                    "item": name,
                    "level": level,
                    "detail": "未安装" + (f"（需要 ≥ {min_version}）" if min_version else ""),
                    "remedy": install_hint,
                })

    return results


def _detect_torch_backend(version: str) -> str:
    """从 torch 版本串识别后端（ROCm / CUDA / CPU-only 构建）。

    ★返回 "CPU" 表示**装了一个用不了的 torch**，不是"可以用 CPU 跑"（§2.14）。
    L1 只用它来标注版本串，不据此放行；真正的放行判据是 L2 的 `probe_device()`。
    """
    if "+rocm" in version:
        return "ROCm"
    if "+cu" in version:
        return "CUDA"
    return "CPU-only 构建"


# ---------------------------------------------------------------------------
# L2 · 动态设备探测（子进程 + 超时）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def probe_device(timeout: int = 90, force: bool = False) -> dict:
    """在子进程中探测设备并返回结果。

    返回：{ok, backend, device_name, torch_version, flash_attn, error}
    - ok: 是否探测成功。**False 意味着终止**（§2.14 GPU-only，没有 CPU 兜底）
    - backend: rocm / cuda（**不再有 cpu** —— 见 §2.14）
    - device_name: 设备名称（如 "AMD Radeon RX 7900 XTX"）
    - torch_version: torch 版本串
    - flash_attn: 是否建议启用 flash attention
    - error: 失败原因（ok=False 时必有）

    ★缓存结果，force=True 时重新探测（供设备切换按钮使用）。
    ★通信方式：子进程将 JSON 结果写入临时文件，避免 stdout 被 torch 污染。
    ★超时按失败处理 —— GPU 状态未知时不允许"先跑起来再说"。
    """
    if force:
        probe_device.cache_clear()

    # 使用 subprocess 运行探测脚本，避免在本进程 import torch/bs_roformer
    # 导致挂死或影响主进程状态
    script = _build_probe_script()

    try:
        # ★子进程环境准备：
        # 1. NUMBA_CACHE_DIR — 避免 librosa 的 numba 装饰器写 site-packages
        #    （目录不可写时静默死循环，不是异常）
        # 2. 结果写入临时文件，避免 stdout 被 torch 污染
        env = os.environ.copy()
        numba_cache_dir = tempfile.mkdtemp(prefix="numba_cache_")
        env["NUMBA_CACHE_DIR"] = numba_cache_dir
        result_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        result_file.close()
        env["PROBE_RESULT_FILE"] = result_file.name

        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            os.unlink(result_file.name)
            import shutil
            shutil.rmtree(numba_cache_dir, ignore_errors=True)
            return {
                "ok": False,
                "backend": "unknown",
                "device_name": "",
                "torch_version": "",
                "flash_attn": False,
                "error": f"探测超时（>{timeout}s）",
            }

        # 从临时文件读取 JSON 结果
        try:
            with open(result_file.name, "r", encoding="utf-8") as f:
                result = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return {
                "ok": False,
                "backend": "unknown",
                "device_name": "",
                "torch_version": "",
                "flash_attn": False,
                "error": f"子进程结果解析失败：{e}",
            }
        finally:
            os.unlink(result_file.name)
            import shutil
            shutil.rmtree(numba_cache_dir, ignore_errors=True)

        return result

    except Exception as e:
        return {
            "ok": False,
            "backend": "unknown",
            "device_name": "",
            "torch_version": "",
            "flash_attn": False,
            "error": str(e),
        }


def _build_probe_script() -> str:
    """构建子进程要运行的探测脚本。

    ★子进程入口必须应用 PLAN §2.8 四条规避，顺序敏感：
    1. 设置 NUMBA_CACHE_DIR 到临时目录
    2. stdout/stderr reconfigure 为 UTF-8（避免 Windows GBK 打印 ⇒ 崩）
    3. 再 import torch/librosa/bs_roformer（顺序颠倒则 1、2 失效）
    4. ffmpeg 的 stderr 单独管道（本探测不涉及 ffmpeg，由调用者处理）

    通信方式：将 JSON 结果写入 PROBE_RESULT_FILE 环境变量指定的临时文件，
    避免 torch 的 stdout 输出污染 JSON 解析。
    """
    return '''
import sys
import json
import os
import tempfile
import warnings

# ★规避 1：设置 NUMBA_CACHE_DIR 到可写临时目录
# 不这样做，librosa 的 numba 装饰器会尝试写 site-packages/__pycache__，
# 目录不可写时 tempfile.TemporaryFile(dir=path) 静默死循环（不是异常）
os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba_cache_")

# ★规避 2：stdout/stderr 重配为 UTF-8
# Windows 默认 GBK，打印 ⇒(U+21D2) 直接崩
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 抑制所有警告（避免污染输出）
warnings.filterwarnings("ignore")

result = {
    "ok": False,
    "backend": "unknown",
    "device_name": "",
    "torch_version": "",
    "flash_attn": False,
    "error": ""
}

try:
    # ★规避 3：再 import 重模块
    import torch
    result["torch_version"] = torch.__version__

    # 步骤 1：torch 导入成功

    # 步骤 2：GPU 可用性
    # ★PLAN §2.14 GPU-only 硬约束：探测不到 GPU 就是终止条件，没有 CPU 兜底。
    # 先按版本串定后端 —— ROCm build 的 torch.cuda.is_available() 同样返回 True，
    # 所以后端只能从 "+rocm" 本地标签判断，不能靠 is_available 的结果反推。
    if "rocm" in torch.__version__:
        result["backend"] = "rocm"
        result["flash_attn"] = False  # ROCm 不启用 flash_attn
    else:
        result["backend"] = "cuda"
        result["flash_attn"] = True  # CUDA 默认尝试启用

    if not torch.cuda.is_available():
        # 区分"装错了后端"和"机器上根本没有 GPU" —— 两者的排查方向完全不同
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
    else:
        result["device_name"] = torch.cuda.get_device_name(0)

        # 步骤 3：import bs_roformer
        from bs_roformer import BSRoformer

        # 步骤 4：一次极小的假推理，捕捉 kernel 层错误（L1 查不到的）
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
        except Exception as e:
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
                except Exception as e2:
                    result["error"] = f"GPU 推理失败（flash_attn=False）：{str(e2)[:200]}"
            else:
                result["error"] = f"GPU 推理失败：{str(e)[:200]}"

except Exception as e:
    result["error"] = str(e)[:500]

# ★通信：将结果写入临时文件（避免 stdout 被 torch 污染）
result_file = os.environ.get("PROBE_RESULT_FILE", "")
if result_file:
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f)
'''


# ---------------------------------------------------------------------------
# L3 · GUI 面板数据结构（不构建 Qt 面板，只提供数据）
# ---------------------------------------------------------------------------

def build_report_rows() -> list[dict]:
    """构建 GUI 面板的数据行。

    返回：list[dict]，每个 dict 包含 item/level/detail/remedy 四个键，
    对应 GUI 表格的四列：项、级别、详情、处置。
    """
    l1_results = check_libs()

    rows = []
    for result in l1_results:
        row = {
            "item": result["item"],
            "level": result["level"],
            "detail": result["detail"],
            "remedy": result["remedy"],
        }
        # torch 的 remedy 是 dict（后端分支），L3 面板按探测结果选取
        if isinstance(row["remedy"], dict):
            # 这里先展示所有后端的安装命令，L3 面板可以进一步筛选
            rows.append(row)
        else:
            rows.append(row)

    return rows


def diagnostic_text() -> str:
    """拼接 L1 + L2 + paths.diagnose() 的完整可复制诊断报告。

    paths 模块懒加载（避免循环导入）。
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Audio Separation 环境诊断报告")
    lines.append("=" * 60)

    # L1 · 静态库检测
    lines.append("")
    lines.append("【L1 静态库检测】")
    for result in check_libs():
        status = result["level"]
        item = result["item"]
        detail = result["detail"]
        remedy_raw = result.get("remedy", "")
        remedy = remedy_raw if not isinstance(remedy_raw, dict) else ""
        lines.append(f"  [{status}] {item}: {detail}")
        if remedy:
            lines.append(f"         修复: {remedy}")

    # L2 · 动态设备探测
    lines.append("")
    lines.append("【L2 动态设备探测】")
    try:
        probe = probe_device()
        lines.append(f"  状态: {'成功' if probe['ok'] else '失败'}")
        lines.append(f"  后端: {probe['backend']}")
        lines.append(f"  设备: {probe['device_name']}")
        lines.append(f"  torch: {probe['torch_version']}")
        lines.append(f"  flash_attn: {probe['flash_attn']}")
        if probe.get("error"):
            lines.append(f"  错误: {probe['error']}")
    except Exception as e:
        lines.append(f"  探测异常: {e}")

    # paths.diagnose() 探测链（懒加载）
    lines.append("")
    lines.append("【路径诊断】")
    try:
        from audio_separation import paths
        lines.append(paths.diagnose())
    except ImportError:
        lines.append("  paths 模块不可用（包安装不完整，src/ 是否在 sys.path 上？）")
    except Exception as e:
        lines.append(f"  路径诊断异常: {e}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
