# -*- coding: utf-8 -*-
"""分离算法（§2.7）：分块 + Hann 窗重叠相加。

## 块位置

```
hop = chunk // (num_overlap + 1)
starts = range(0, total - chunk + 1, hop)      # 逐块推进
若末块盖不到文件尾 → 追加 total - chunk      # ★必须补，否则最后 <hop 秒永远没被推理
```

`num_overlap=2` ⇒ `hop = chunk/3`，即相邻块重叠 2/3 块长。
60 s 切片（chunk 11.0 s / hop 3.667 s）⇒ **15 块** / 14 个内部重叠边界
（14 块按 hop 等步推进，第 15 块是贴右边界的补位块）。

## ★ 加窗与归一化：为什么分母是 `Σ(w)` 而不是 `Σ(w²)`

设第 b 块在样本 i 上的窗值为 `w(i-s_b)`，合成结果

```
out[i] = Σ_b y_b[i]·w(i-s_b) / Σ_b w(i-s_b)
```

这是**加权平均**：模型输出等于输入（`y=x`）时 `out[i] = x[i]·Σw/Σw = x[i]`，
**逐样本精确重建**（本模块 `test_reconstruction_is_exact` 在 CPU 上验证过：
内部增益恒为 1.0000）。

如果分母换成 `Σ(w²)` 而分子仍是 `Σ(y·w)`，`y=x` 时得到
`x·Σw/Σw² = 1.3333·x` —— **整轨偏大 4 dB**。
只有当分子也同步加窗（`Σ(y·w²)/Σ(w²)`）才精确重建，那是「分析窗 + 合成窗
各加一次」的约定；本模型只作用于**未加窗**的原始块，合成侧是唯一一次加窗，
所以用 `Σw/Σw`。

> PLAN.md §2.7 原先写的 `wsum += window**2` 是上述错误组合，本次实现已修正
> PLAN 并在此说明理由。参考脚本 `tools/bs_roformer_revive_separate.py` 用的
> 就是本文的正确公式，所以回归基线仍然可比。

## `periodic=False` 不能改

参考脚本注释原文：periodic=True 会引入周期性断点。`hann_window(N)`
（periodic=True）等价于 `hann_window(N+1)[:-1]`，两端不归零；
分块拼接时每块边界都会留下一个小台阶，60 s 切片有 13 个内部边界 ⇒ 听得出来。

## ★ 看门狗

ROCm 算子挂死时（实测）Python 层**无法中断** —— `KeyboardInterrupt` 不生效，
`torch` 的 C++ 调用栈不返回。所以看门狗由独立线程 `os._exit(3)` 强杀，
**只能在推理子进程里用**（§2.3），GUI 主进程绝不能开。实现见 `watchdog.py`。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .watchdog import BlockWatchdog

LogFn = Callable[..., None]
ProgressFn = Callable[["BlockProgress"], None]
CancelFn = Callable[[], bool]


class Cancelled(Exception):
    """用户在分离途中点了取消。"""


class _Model(Protocol):
    """`build_model()` 的产物在本模块只用到一个前向调用。"""

    def __call__(self, x: "object") -> "object":  # pragma: no cover - 仅为类型标注
        ...


# ══════════════════════════════════════════════════════════════════════════════
# 纯计算（不碰 torch，可在快速测试里直接断言）
# ══════════════════════════════════════════════════════════════════════════════
def chunk_starts(total: int, chunk: int, num_overlap: int) -> list[int]:
    """算出所有块的起始样本下标。

    Args:
        total: 音频总样本数。
        chunk: 块长（模型固定 485100）。
        num_overlap: 重叠块数（模型固定 2）。

    Returns:
        升序、不重复的起始下标列表。**保证最后一块盖到文件尾**。
    """
    if total <= 0:
        return []
    if num_overlap < 0:
        raise ValueError(f"num_overlap 不能为负：{num_overlap}")
    if total <= chunk:
        return [0]

    hop = max(1, chunk // (num_overlap + 1))
    starts = list(range(0, total - chunk + 1, hop))
    # ★ 末块盖不满整段时补一个贴右边界的位置，否则最后 (total % hop) 个样本
    #   永远没有任何窗口覆盖 → wsum 恒 0 → 被 1e-8 兜底成静音。
    if starts[-1] + chunk < total:
        starts.append(total - chunk)
    return starts


def hann_window(chunk: int) -> np.ndarray:
    """`float64` 的对称 Hann 窗（`periodic=False`，两端归零）。

    ★ 不用 `torch.hann_window` 是为了让本函数**不依赖 torch**——
      分块/加窗逻辑要能在没装 torch 的环境下被测试直接验证。
    """
    n = int(chunk)
    if n <= 0:
        raise ValueError(f"chunk 必须为正：{chunk}")
    if n == 1:
        return np.ones(1, dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))


def window_weights(chunk: int, num_overlap: int, total: int) -> np.ndarray:
    """逐样本累计到的窗权重 `Σw`，也就是归一化的**分母**。

    诊断价值：
    - 内部应当**恒定**（这是加权平均能精确重建的前提）；
    - 接近 0 的位置 = 该样本几乎没有窗口覆盖，输出完全由模型在补零区
      的行为决定，**不可信**（`separate_vocals` 内部恒定的首/末样本就是这种）。
    """
    w = hann_window(chunk)
    acc = np.zeros(total, dtype=np.float64)
    for s in chunk_starts(total, chunk, num_overlap):
        n = min(chunk, total - s)
        acc[s:s + n] += w[:n]
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# 进度 / 看门狗
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class BlockProgress:
    """单块完成后的进度快照。"""

    done: int
    total: int
    start_sec: float
    end_sec: float
    elapsed_sec: float
    eta_sec: float

    def as_line(self) -> str:
        return (
            f"  块 {self.done}/{self.total}  "
            f"{self.start_sec:7.1f}s–{self.end_sec:7.1f}s  "
            f"单块 {self.elapsed_sec / self.done:5.1f}s  "
            f"已用 {self.elapsed_sec / 60:.1f}min  "
            f"ETA {self.eta_sec / 60:.1f}min"
        )


def _noop(*_a: object, **_k: object) -> None:
    """默认空实现。"""


# ══════════════════════════════════════════════════════════════════════════════
# 分离
# ══════════════════════════════════════════════════════════════════════════════
def separate_vocals(
    model: Callable[[np.ndarray], np.ndarray],
    wav: np.ndarray,
    sample_rate: int,
    *,
    chunk: int,
    num_overlap: int,
    device: str = "cuda",
    dtype: str = "float32",
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    block_timeout_sec: float = 0.0,
    log: LogFn = _noop,
) -> np.ndarray:
    """分块推理，返回人声轨 `(total, 2)` float32。

    ★ 单 stem 模型：只输出人声。instrumental 由调用方算残差
      `wav - vocals`（§2.7），不要在这里造第二份显存。

    Args:
        model: 已 `eval()` 且已 `.to(device)` 的模型。
        wav: `(total, 2)` float32 输入。
        sample_rate: 只用于日志换算秒数。
        chunk: 块长，取自 config `audio.chunk_size`。
        num_overlap: 重叠块数，取自 config `inference.num_overlap`。
        device: 必须是 ``cuda``（ROCm 下也是这个名）—— PLAN §2.14 GPU-only。
            这里的 ``.cpu()`` 是**把结果搬回主机**做加窗累加，不是在 CPU 上做推理。
        dtype: `float32` 或 `bfloat16`。★ ROCm 上 STFT 用 bf16 会掉精度（§2.5）。
        on_progress: 每块完成后回调（`BlockProgress`）。★ 在主线程调用，不要在里面做重活。
        should_cancel: 返回 True 则在块边界抛 `Cancelled`。★ 只能块边界取消——
          一块 11 s 的前向在 ROCm 挂死时无法中断。
        block_timeout_sec: >0 时启用看门狗（★仅限子进程）。
        log: 日志回调。

    Returns:
        `(total, 2)` float32 人声轨。
    """
    import torch

    total = int(wav.shape[0])
    if total == 0:
        raise ValueError("wav 是空的")
    if wav.ndim != 2 or wav.shape[1] != 2:
        raise ValueError(f"wav 必须是 (total, 2)，实际 {wav.shape}")

    torch_dtype = getattr(torch, dtype)
    starts = chunk_starts(total, chunk, num_overlap)
    # ★ win 必须一维 (chunk,)：与一维的 wsum 相加时，任何二维形状都会被广播成 (n, n)
    win = torch.from_numpy(hann_window(min(chunk, total))).to(device=device, dtype=torch_dtype)

    vocals = np.zeros((total, 2), dtype=np.float32)
    wsum = np.zeros(total, dtype=np.float64)

    watchdog = BlockWatchdog(block_timeout_sec).start() if block_timeout_sec > 0 else None
    t0 = time.monotonic()

    try:
        # ★★★ no_grad 是本函数能否跑完的唯一前提，不是优化。
        # 160 M 参数 × 11 s 音频的前向会产生数 GiB 的中间量；不开 no_grad 时 autograd
        # 把**每一块**的整张计算图都挂在 output 上，15 块累积到 53 GiB（实测：
        # "GPU 0 has a total capacity of 23.98 GiB of which 0 bytes is free.
        #  Of the allocated memory 53.52 GiB is allocated by PyTorch"）。
        # 而 allocator 在满显存下的回收路径极慢，表现为"跑很久也不报错的假死"。
        # 用 no_grad 而不是 inference_mode：参考脚本用的就是它，本机 ROCm 构建已验证。
        with torch.no_grad():
            for i, s in enumerate(starts):
                if should_cancel is not None and should_cancel():
                    raise Cancelled(f"第 {i + 1}/{len(starts)} 块前收到取消请求")

                seg = wav[s:s + chunk]
                n = int(seg.shape[0])
                if n < chunk:
                    # 末块补零：模型要求定长输入。补零区不计入输出（下面只取 [:n]）。
                    seg = np.pad(seg, ((0, chunk - n), (0, 0)))

                if watchdog:
                    watchdog.beat(f"块 {i + 1}/{len(starts)} 前向")

                x = torch.from_numpy(np.ascontiguousarray(seg.T)).unsqueeze(0)
                x = x.to(device=device, dtype=torch_dtype)
                y = model(x)  # (1, 1, 2, chunk)

                if watchdog:
                    watchdog.beat(f"块 {i + 1}/{len(starts)} 回传 CPU")

                # 单 stem：(1,1,2,chunk) → squeeze 两维 → (2,chunk) → 转置 (chunk,2)
                block = y.squeeze(0).squeeze(0).T.float().cpu().numpy()

                # ★ 立刻交还显存。y 及其整条中间链在下一轮绑定前一直被引用着；
                #   单块前向的中间量在 24 GiB 卡上已接近天花板，不能等自然回收。
                del x, y

                w = win[:n].float().cpu().numpy().astype(np.float64)
                # ★ w 是 (n,)、vocals 是 (n, 2)：加轴不可省，(n,2)*(n,) 会直接报错
                vocals[s:s + n] += block[:n] * w[:, None]
                wsum[s:s + n] += w

                done = i + 1
                prog = BlockProgress(
                    done=done,
                    total=len(starts),
                    start_sec=s / sample_rate,
                    end_sec=(s + n) / sample_rate,
                    elapsed_sec=time.monotonic() - t0,
                    eta_sec=(time.monotonic() - t0) / done * (len(starts) - done),
                )
                log(prog.as_line())
                if on_progress is not None:
                    on_progress(prog)
    finally:
        if watchdog is not None:
            watchdog.stop()

    # ★ 分母兜底 1e-8：Hann 窗两端恰好为 0，首/末样本的 wsum 可能真的是 0。
    vocals /= np.maximum(wsum, 1e-8).astype(np.float32)[:, None]
    return vocals


__all__ = [
    "Cancelled",
    "BlockProgress",
    "BlockWatchdog",
    "chunk_starts",
    "hann_window",
    "window_weights",
    "separate_vocals",
]
