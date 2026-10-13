# -*- coding: utf-8 -*-
"""模型构建与权重校验（PLAN §2.2）。

下面 ``build_model`` 里的每一步都带着「为什么」，一条都不能省。
参考脚本 ``tools/bs_roformer_revive_separate.py`` 是它们的出处；
删掉任何一条的后果不是报错，而是**静默地输出错位的人声**。
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

from audio_separation import paths
from audio_separation.core.config import ConfigError, describe_config, load_config, load_config_by_key

if TYPE_CHECKING:
    # ★只在类型检查时需要 torch。运行期不 import，理由有二：
    #   1) §2.14 之后 torch 是"必须存在且必须是 GPU 构建"的硬依赖，
    #      让本模块能被 import 只用来读元数据（list_models / describe），
    #      可以在 torch 缺失或坏掉时仍给出可用的诊断信息；
    #   2) 与 pipeline.py / separator.py 的延迟导入体例一致，
    #      快速测试集不必为断言一个参数默认值就拖进 1.5 s 的 torch 导入。
    # from __future__ import annotations 已让签名里的 torch.nn.Module 变成字符串。
    import torch

Logger = Callable[[str], None]


def build_model(
    cfg: dict,
    *,
    device: str = "cuda",
    flash_attn: bool | None = None,
    log: Logger = print,
) -> torch.nn.Module:
    """按配置构造模型并加载权重。

    Args:
        cfg: ``load_config()`` 的返回值。
        device: 目标设备。★只支持 ``cuda``（PLAN §2.14 GPU-only）——
            ROCm wheel 的 torch 也用这个设备串，所以两种 GPU 后端无需分支。
        flash_attn: 是否启用 flash attention。``None`` 表示按 PLAN §2.2 第 1 条
            保守取 False——§2.6 的 L2 探测没跑完时必须走这条路。
        log: 日志回调。

    Returns:
        已 ``eval()``、已转 float32、已搬到 ``device`` 的模型。

    Raises:
        ConfigError: 配置里缺必要字段。
        ValueError: ``device`` 不是 ``cuda``。
        RuntimeError: 权重与模型键名或形状对不上。

    ★``torch.load(map_location="cpu")`` 不是"用 CPU 跑模型"：610 MB 权重必须先
    经内存才能进显存，torch 没有"直接加载到显存"的路径。禁止的是把**前向计算**
    放在 CPU 上，而这里的 ``map_location`` 只影响加载终点（下面立刻 ``.to(device)``）。
    """
    if device != "cuda":
        raise ValueError(
            f"不支持的设备「{device}」：本工具只做 GPU 推理（PLAN §2.14）。"
        )

    import torch  # 延迟导入：见文件头的 TYPE_CHECKING 说明
    from bs_roformer import BSRoformer

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ConfigError("配置里没有 model 小节")

    mcfg = dict(model_cfg)
    for key in ("freqs_per_bands", "multi_stft_resolutions_window_sizes"):
        if key in mcfg and not isinstance(mcfg[key], tuple):
            mcfg[key] = tuple(mcfg[key])

    # ① flash_attn —— 默认 False。ROCm 上第三方 CUDA kernel 不可用；
    #    只有探测确认后端为 CUDA 时才允许打开。探测未完成 = 不打开。
    mcfg["flash_attn"] = bool(flash_attn) if flash_attn is not None else False
    log(f"  · flash_attn = {mcfg['flash_attn']}")

    # ② num_residual_streams —— revive 权重的键名里没有 `.branch.` 这一层，
    #    多流版本构造出来的参数名对不上权重。
    mcfg["num_residual_streams"] = 1

    # ③ BS-RoFormer 升级会加新参数，而 revive 权重跟不上。静默丢弃会让
    #    「版本不兼容」变成一次难查的权重不匹配，所以把丢掉的键打出来。
    allowed = set(inspect.signature(BSRoformer.__init__).parameters) - {"self"}
    dropped = sorted(set(mcfg) - allowed)
    if dropped:
        log(f"  · 忽略 config 中 BSRoformer 不接受的键：{dropped}")
    mcfg = {k: v for k, v in mcfg.items() if k in allowed}

    model = BSRoformer(**mcfg)

    # ④ to_value_residual_mix 是 lucidrains 后期加的模块，revive 权重里没有。
    #    不置 None 的话 load_state_dict 会报出多出来的键。
    for submodule in model.modules():
        if hasattr(submodule, "to_value_residual_mix"):
            submodule.to_value_residual_mix = None

    # ⑤ Lightning checkpoint 的权重可能裹在 state_dict / model_state_dict 里
    ckpt = torch.load(cfg["_weights"], map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    # ⑥ 严格比对：hyper-connections 在模型侧引入了 .branch. 这一层，
    #    权重侧没有，所以先剥掉再比。键集合与形状双双对齐才算过。
    pairs = []
    extra = []
    remapped_names = set()
    for own_key, tensor in model.state_dict().items():
        key = own_key.replace(".branch.", ".")
        remapped_names.add(key)
        if key not in sd:
            extra.append(key)
            continue
        if tensor.shape != sd[key].shape:
            raise RuntimeError(
                f"权重形状不匹配 {key}：模型 {tuple(tensor.shape)} vs 权重 {tuple(sd[key].shape)}"
            )
        pairs.append((tensor, sd[key]))
    unused = sorted(set(sd) - remapped_names)
    if extra or unused:
        raise RuntimeError(
            f"键名对不上：模型有而权重无={extra[:8]}，权重有而模型无={unused[:8]}"
        )
    with torch.no_grad():
        for tensor, value in pairs:
            tensor.copy_(value)
    log(f"  · 已加载 {len(pairs)} 个参数张量，键名与形状全部对齐")

    # ⑦ fp32：ROCm 上 bf16 会让 STFT 掉精度。这不是保守，是实测。
    # 先转 dtype 再搬设备：转换在来源端做，避免为 dtype 转换额外占一份显存。
    return model.eval().to(torch.float32).to(device)


def check_model(model_key: str, *, device: str = "cuda", flash_attn: bool | None = None) -> dict:
    """加载模型并返回一段可打印的描述，用于人工排查（PLAN §2.2 `__main__ --check`）。"""
    cfg = load_config_by_key(model_key)
    log_lines: list[str] = []

    def _log(message: str) -> None:
        log_lines.append(message)
        print(message)

    model = build_model(cfg, device=device, flash_attn=flash_attn, log=_log)
    total = sum(p.numel() for p in model.parameters())
    return {
        "key": model_key,
        "weights": cfg["_weights"],
        "parameters": total,
        "rows": describe_config(cfg),
        "log": log_lines,
        "model": model,
    }


def list_models() -> list[paths.ModelInfo]:
    """列出成对的 yaml + ckpt（探测本身不做任何 IO 之外的校验）。"""
    return paths.list_models()


def __main__(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m audio_separation.core.model",
        description="构建模型并逐张量校验权重是否对得上（人工排查用）",
    )
    parser.add_argument("--check", metavar="KEY", help="要校验的模型键，如 v2 / v3")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda"],
        help="目标设备，只支持 cuda（PLAN §2.14 GPU-only，无 CPU 模式）",
    )
    parser.add_argument("--list", action="store_true", help="只列出可用模型")
    args = parser.parse_args(argv)

    models = list_models()
    if args.list or not args.check:
        print(f"可用模型：{[m.key for m in models] or '（无）'}")
        return 0 if models else 1

    try:
        info = check_model(args.check, device=args.device)
    except (ConfigError, ValueError, RuntimeError) as exc:
        print(f"[{args.check}] 失败：{exc}", file=sys.stderr)
        return 1

    print(f"[{info['key']}] {info['weights']}")
    print(f"  参数量：{info['parameters']:,}")
    for label, value in info["rows"]:
        print(f"    {label:<20} {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys

    raise SystemExit(__main__(_sys.argv[1:]))
