"""慢集共享的权重缓存（PLAN §6.5 替代已删除的 `model_v2` / `model_v3` fixture）。

## 为什么需要它

`test_pipeline.py` 与 `test_regression_baseline.py` 都要在真实 GPU 上跑推理，
各自至少要加载一次 610 MB 权重 + 2.9 s 构建。两个测试各加载一次就是多付一次。

## 为什么不用 fixture

conftest 里的 fixture 只能靠参数把 key 传进去，而"用不用缓存"是**测试自己**的事
（`test_..._rejects_a_broken_weight` 故意要拿一个坏模型，不能被缓存串味）。
函数式 helper 让每个测试显式写 `load_cached("v2")`，意图一眼可见。

## 进程边界

它缓存的是**当前 pytest 进程**内的模型。跨文件不共享 —— 这没关系：
GPU 推理本来就要求"一次进程内只建一次模型"，跨进程复用会让显存状态变得难推理。
"""
from __future__ import annotations

from typing import Any

_CACHE: dict[str, Any] = {}


def load_cached(model_key: str, *, device: str = "cuda", flash_attn: bool = False) -> Any:
    """构建并缓存 `model_key` 的模型；同一进程内第二次调用直接返回同一个实例。

    Args:
        model_key: `"v2"` / `"v3"`。
        device: 只允许 `"cuda"`（PLAN §2.14 GPU-only）。
        flash_attn: L2 探测的结论；未探测时传 False。

    Returns:
        已 `.eval()` 且在目标设备上的 `nn.Module`。

    Raises:
        ValueError: `device` 不是 cuda，或 `model_key` 为空。
    """
    if device != "cuda":
        raise ValueError(f"本工具只做 GPU 推理，device={device!r} 非法（PLAN §2.14）")
    if not model_key:
        raise ValueError("model_key 不能为空")

    hit = _CACHE.get(model_key)
    if hit is not None:
        return hit

    from audio_separation.core.config import load_config_by_key
    from audio_separation.core.model import build_model

    model = build_model(
        load_config_by_key(model_key), device=device, flash_attn=flash_attn
    )
    _CACHE[model_key] = model
    return model


def clear() -> None:
    """清空缓存。给需要验证「重复加载」本身是否正确的测试用。"""
    _CACHE.clear()
