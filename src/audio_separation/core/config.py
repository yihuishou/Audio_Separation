# -*- coding: utf-8 -*-
"""模型配置加载与交叉校验（PLAN §2.1）。

★核心立场：**Python 侧不硬编码任何超参数**。
v2 与 v3 的全部差异都只是 yaml 里的数据，代码里一个分支都没有——
所以换模型版本只需要换 yaml + ckpt，不用改代码、也不用重新测逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_separation import paths

#: revive 权重只在 44100 Hz 下训练，改采样率等于无声地降质
TRAINED_SAMPLE_RATE = 44100

#: yaml 必须具备的四个小节
REQUIRED_SECTIONS = ("audio", "model", "training", "inference")


class ConfigError(Exception):
    """配置缺失、结构不符或交叉校验不通过。"""


class _YamlLoader(yaml.SafeLoader):
    pass


#: yaml 文件里显式写了 `!!python/tuple`（PyTorch 导出的配置惯例），
#: 而 SafeLoader 默认不认识这个标签，直接 load 会抛 ConstructorError。
def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


_YamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def _dig(cfg: dict, dotted: str) -> Any:
    """按 ``"model.dim_freqs_in"`` 取值；任一层缺失都返回 ``None``。"""
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def validate_config(cfg: dict) -> list[str]:
    """逐条执行 §2.1 的交叉校验，返回**全部**不通过项的说明（而非遇错即停）。

    一次报完是为了让用户改一遍就能改对；只报第一条会让人来回试四次。

    Args:
        cfg: 原始 yaml 字典。

    Returns:
        问题说明列表；空列表表示全部通过。
    """
    problems: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            problems.append(f"缺少小节 {section!r}")
    if problems:
        return problems

    n_fft = _dig(cfg, "audio.n_fft")
    dim_freqs_in = _dig(cfg, "model.dim_freqs_in")
    sample_rate = _dig(cfg, "audio.sample_rate")
    audio_dim_t = _dig(cfg, "audio.dim_t")
    inference_dim_t = _dig(cfg, "inference.dim_t")
    chunk_size = _dig(cfg, "audio.chunk_size")
    hop_length = _dig(cfg, "audio.hop_length")

    # ① 频率维：STFT 给出的频点数必须正好等于模型要吃的频点数
    if n_fft is not None and dim_freqs_in is not None:
        expected = n_fft // 2 + 1
        if dim_freqs_in != expected:
            problems.append(
                f"model.dim_freqs_in={dim_freqs_in}，但 audio.n_fft={n_fft} 对应 {expected}；"
                "频点数错位会让 STFT 输出与模型输入对不上"
            )

    # ② 采样率
    if sample_rate is not None and sample_rate != TRAINED_SAMPLE_RATE:
        problems.append(
            f"audio.sample_rate={sample_rate}，但权重只在 {TRAINED_SAMPLE_RATE} Hz 下训练；"
            "强行换采样率会得到变调式的劣化结果"
        )

    # ③ 推理用的时间维必须与音频配置一致
    if audio_dim_t is not None and inference_dim_t is not None and audio_dim_t != inference_dim_t:
        problems.append(
            f"audio.dim_t={audio_dim_t} 与 inference.dim_t={inference_dim_t} 不一致；"
            "推理与训练的时间维对不上"
        )

    # ④ ★分块长度 = hop_length × (dim_t − 1)，不是 × dim_t。
    #    少一格是 STFT 边界对齐的必然结果：dim_t 个频率帧跨 dim_t−1 个采样步长。
    #    ★这个 −1 写错不会报错，只会让模型静默输出错位的人声，所以必须在校验里钉死。
    if None not in (chunk_size, hop_length, audio_dim_t):
        expected_chunk = hop_length * (audio_dim_t - 1)
        if chunk_size != expected_chunk:
            wrong = hop_length * audio_dim_t
            problems.append(
                f"audio.chunk_size={chunk_size}，但 audio.hop_length×(audio.dim_t−1)"
                f"={hop_length}×{audio_dim_t - 1}={expected_chunk}"
                f"（若误写成 ×dim_t 则为 {wrong}）；写错会让模型静默输出错位的人声"
            )

    return problems


def load_config(ckpt_path: Path | str) -> dict:
    """读取 ckpt 同目录下的同名 ``.yaml``，校验通过后返回配置字典。

    返回的字典额外注入两个下划线开头的键，供下游定位文件用：
    ``_dir``（配置文件所在目录）与 ``_weights``（对应 ckpt 路径）。

    Raises:
        ConfigError: yaml 缺失、结构不符、或交叉校验不通过。
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise ConfigError(f"权重文件不存在：{ckpt_path}")

    yaml_path = ckpt_path.with_suffix(".yaml")
    if not yaml_path.is_file():
        raise ConfigError(f"权重 {ckpt_path.name} 没有配套的 yaml：{yaml_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.load(fh, Loader=_YamlLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"yaml 解析失败（{yaml_path}）：{exc}") from exc

    if not isinstance(cfg, dict):
        raise ConfigError(f"{yaml_path.name} 顶层不是字典，而是 {type(cfg).__name__}")

    problems = validate_config(cfg)
    if problems:
        detail = "\n  - ".join(problems)
        raise ConfigError(f"{yaml_path.name} 配置校验未通过：\n  - {detail}")

    cfg["_dir"] = str(yaml_path.parent)
    cfg["_weights"] = str(ckpt_path)
    return cfg


def load_config_by_key(model_key: str) -> dict:
    """按 ``"v2"`` / ``"v3"`` 这样的键加载配置。

    Raises:
        ConfigError: 该键在 ``models/bs_roformer_revive/`` 下没有成对的 yaml+ckpt。
    """
    for info in paths.list_models():
        if info.key == model_key:
            return load_config(info.ckpt_path)
    available = ", ".join(m.key for m in paths.list_models()) or "（一个都没有）"
    raise ConfigError(f"没有名为 {model_key!r} 的模型；现有的有：{available}")


def describe_config(cfg: dict) -> list[tuple[str, str]]:
    """产出 §2.11 只读参数面板要显示的 (标签, 值) 列表。

    这些值全部来自 yaml，界面上不可编辑——改了就会和权重对不上。
    """
    audio = cfg.get("audio", {})
    model = cfg.get("model", {})
    inference = cfg.get("inference", {})
    return [
        ("采样率", f"{audio.get('sample_rate')} Hz"),
        ("分块长度", f"{audio.get('chunk_size')}（≈ {audio.get('chunk_size', 0) / audio.get('sample_rate', 1):.1f} s）"),
        ("帧移 hop_length", str(audio.get("hop_length"))),
        ("时间维 dim_t", str(audio.get("dim_t"))),
        ("频域维 dim", str(model.get("dim"))),
        ("层数 depth", str(model.get("depth"))),
        ("分离目标", ", ".join(cfg.get("training", {}).get("instruments", []))),
        ("目标乐器", str(cfg.get("training", {}).get("target_instrument"))),
        ("重叠块数 num_overlap", str(inference.get("num_overlap"))),
        ("STFT n_fft / hop", f"{audio.get('n_fft')} / {audio.get('hop_length')}"),
    ]


if __name__ == "__main__":  # pragma: no cover
    for info in paths.list_models():
        config = load_config(info.ckpt_path)
        print(f"[{info.key}] {info.yaml_path.name}")
        for label, value in describe_config(config):
            print(f"    {label:<20} {value}")
