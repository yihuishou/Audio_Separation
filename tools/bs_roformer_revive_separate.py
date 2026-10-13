#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bs-roformer-revive-v3 音轨分离（人声/其他全部音效）。

用途
----
使用 `models/bs_roformer_revive/bs_roformer_revive_v3_by_unwa.ckpt` 单 stem 模型，
分离出**人声轨**和**其他全部音效轨**（原轨减去人声）。

与 viperx-1297 工具的区别
-------------------------
- revive 是**单 stem**（num_stems=1，只输出人声），不是 4 stem
- instrumental 轨 = 原轨 - 人声轨（残差），不是模型直接输出
- 权重是 PyTorch Lightning `.ckpt` 格式（非 safetensors）
- 模型配置：dim=512, depth=12（比 viperx 的 dim=384, depth=8 更大）

用法
----
  python tools/bs_roformer_revive_separate.py -i input.mp3
  python tools/bs_roformer_revive_separate.py -i in.mp3 -o output/separated
  python tools/bs_roformer_revive_separate.py -i in.mp3 --dry-run

产物
----
  <stem>.vocals.wav         人声
  <stem>.instrumental.wav   其他全部音效（原轨 - 人声）
  <stem>.mix_check.wav      人声+音效相加，验证互补性
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# ★ 必须在 import librosa / bs_roformer 之前设好。librosa 的 numba 装饰器在
#   import 期会去 site-packages 的 __pycache__ 写缓存文件，本机该目录不可写，
#   `tempfile.TemporaryFile(dir=path)` 会**静默死循环**。指到 TEMP 就正常。
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(os.environ.get("TEMP", REPO / "output")) / "numba_cache"),
)
# ★ 本机 stdout 默认 GBK，重定向到文件时中文/箭头会 UnicodeEncodeError
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import yaml

from audio_ai.paths import find_ff, require_dir

MODEL_SUBDIR = "bs_roformer_revive"
WEIGHTS_FILE = "bs_roformer_revive_v3_by_unwa.ckpt"
CONFIG_FILE = "bs_roformer_revive_v3_by_unwa.yaml"


def log(*a):
    print(*a, flush=True)


# ────────────────────────────────────────────── 配置读取
class _YamlLoader(yaml.SafeLoader):
    """★ config.yaml 里的 tuple 用 `!!python/tuple` 标注，SafeLoader 直接报
    ConstructorError。SafeLoader + 显式构造器最安全（不走 unsafe load）。"""


_YamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def load_config() -> dict:
    model_dir = require_dir(REPO / "models" / MODEL_SUBDIR, "bs-roformer-revive 模型")
    cfg = yaml.load(
        (model_dir / CONFIG_FILE).read_text(encoding="utf-8"), Loader=_YamlLoader
    )
    cfg["_dir"] = model_dir
    cfg["_weights"] = model_dir / WEIGHTS_FILE
    return cfg


def build_model(cfg: dict) -> torch.nn.Module:
    """按 config 构造模型并加载 .ckpt 权重。

    revive 是单 stem 人声分离模型：输出人声轨，instrumental = 原轨 - 人声。
    """
    from bs_roformer import BSRoformer

    mcfg = dict(cfg["model"])
    mcfg["freqs_per_bands"] = tuple(mcfg["freqs_per_bands"])
    mcfg["multi_stft_resolutions_window_sizes"] = tuple(
        mcfg["multi_stft_resolutions_window_sizes"]
    )
    # ROCm 上 flash-attn（第三方 CUDA kernel）不可用
    mcfg["flash_attn"] = False
    # revive 权重键名无 .branch. 层，必须 num_residual_streams=1
    mcfg["num_residual_streams"] = 1

    import inspect
    allowed = set(inspect.signature(BSRoformer.__init__).parameters) - {"self"}
    dropped = sorted(set(mcfg) - allowed)
    if dropped:
        log(f"  · 忽略 config 中不被 BSRoformer 接受的键: {dropped}")
    mcfg = {k: v for k, v in mcfg.items() if k in allowed}

    model = BSRoformer(**mcfg)

    # ★ `to_value_residual_mix` 是 lucidrains 后来加的，ZFTurbo/revive 权重里没有
    for m in model.modules():
        if hasattr(m, "to_value_residual_mix"):
            m.to_value_residual_mix = None

    # 加载 .ckpt（PyTorch Lightning 格式）
    ckpt = torch.load(cfg["_weights"], map_location="cpu", weights_only=False)
    # Lightning checkpoint 的权重可能在 'state_dict' 或 'model_state_dict' 里
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    # 剥掉模型侧 hyper-connections 引入的 .branch.（权重侧本来就没有）
    pairs = []
    extra = []
    remapped_names = set()
    for own_k, t in model.state_dict().items():
        k = own_k.replace(".branch.", ".")
        remapped_names.add(k)
        if k not in sd:
            extra.append(k)
            continue
        if t.shape != sd[k].shape:
            raise RuntimeError(
                f"权重形状不匹配 {k}: 模型 {tuple(t.shape)} vs 权重 {tuple(sd[k].shape)}"
            )
        pairs.append((t, sd[k]))
    unused = sorted(set(sd) - remapped_names)
    if extra or unused:
        raise RuntimeError(
            f"键名对不上：模型有而权重无={extra[:8]}，权重有而模型无={unused[:8]}"
        )
    with torch.no_grad():
        for t, v in pairs:
            t.copy_(v)
    log(f"  · 已加载 {len(pairs)} 个参数张量，键名与形状全部对齐")
    return model.eval()


# ────────────────────────────────────────────── 音频 IO（ffmpeg 解码）
def decode_audio(ff: Path, src: Path, sr: int):
    """解码成 (n, 2) float32 numpy。"""
    import subprocess

    cmd = [str(ff), "-v", "error", "-i", str(src),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
    log(f"  · ffmpeg 解码 → {sr} Hz 立体声 f32le")
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    log(f"  · 解码得到 {wav.shape[0]} 样本 = {wav.shape[0]/sr:.1f}s")
    return wav


def write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    """写 16bit PCM WAV。用 scipy.io.wavfile（避免 soundfile libsndfile 1.2.2 bug）。"""
    import scipy.io.wavfile as sfw

    pcm = np.clip(wav, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    tmp = path.with_suffix(path.suffix + ".tmp")
    sfw.write(str(tmp), sr, pcm)
    os.replace(tmp, path)


# ────────────────────────────────────────────── 分离（分块 + 重叠相加）
class _Watchdog(threading.Thread):
    """看门狗：主线程卡在某个 CUDA 算子上时 Python 没法中断它，
    只能由本线程检测「距上次进度超过 N 秒」并带诊断信息强行退出。"""

    def __init__(self, limit: float):
        super().__init__(daemon=True)
        self.limit = limit
        self.stop = threading.Event()
        self.last = time.time()
        self.stage = "启动"

    def beat(self, stage: str) -> None:
        self.stage = stage
        self.last = time.time()

    def run(self) -> None:
        while not self.stop.wait(1.0):
            idle = time.time() - self.last
            if idle < self.limit:
                continue
            mem = ""
            try:
                mem = (f"，显存 已分配 {torch.cuda.memory_allocated()/2**30:.2f} GiB"
                       f" / 峰值 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
            except Exception:
                pass
            print(f"\n[超时] 阶段「{self.stage}」已 {idle:.0f}s 无进度"
                  f"（阈值 {self.limit:.0f}s）{mem}\n"
                  f"       ROCm 算子挂死无法从 Python 侧中断，强制退出。\n",
                  flush=True)
            os._exit(3)


@torch.no_grad()
def separate_vocals(model, wav: np.ndarray, sr: int, device: str,
                    chunk: int, num_overlap: int,
                    block_timeout: float = 0.0) -> np.ndarray:
    """按 config 的 chunk_size 切块推理，重叠区用 Hann 窗加权相加。
    
    单 stem 模型：输出人声轨。
    """
    total = wav.shape[0]
    hop = max(1, chunk // (num_overlap + 1))
    starts = list(range(0, max(1, total - chunk + 1), hop))
    if not starts:
        starts = [0]
    if starts[-1] + chunk < total:
        starts.append(total - chunk)

    vocals_out = np.zeros((total, 2), dtype=np.float32)
    wsum = np.zeros((1, total, 1), dtype=np.float32)

    win = torch.hann_window(chunk, periodic=False).to(device).view(1, -1, 1)

    wd = _Watchdog(block_timeout) if block_timeout > 0 else None
    if wd:
        wd.start()

    t0 = time.time()
    for i, s in enumerate(starts):
        seg = wav[s:s + chunk]
        n = seg.shape[0]
        if n < chunk:
            seg = np.pad(seg, ((0, chunk - n), (0, 0)))

        if wd:
            wd.beat(f"块 {i+1}/{len(starts)} 前向")
        x = torch.from_numpy(seg.T).unsqueeze(0).to(device)   # (1,2,chunk)
        y = model(x)                                          # (1,1,2,chunk)

        if wd:
            wd.beat(f"块 {i+1}/{len(starts)} 回传 CPU")
        # 单 stem 模型：(1,1,2,chunk) → squeeze 两次 → (2,chunk) → 转置成 (chunk,2)
        y = y.squeeze(0).squeeze(0).T.cpu().numpy()           # (chunk,2)
        vocals_out[s:s + n] += y[:n] * win[0, :n].cpu().numpy()
        wsum[0, s:s + n] += win[0, :n].cpu().numpy()

        done = i + 1
        el = time.time() - t0
        log(f"  块 {done}/{len(starts)}  {s/sr:7.1f}s–{(s+n)/sr:7.1f}s  "
            f"单块 {el/done:5.1f}s  已用 {el/60:.1f}min  "
            f"ETA {el/done*(len(starts)-done)/60:.1f}min")

    if wd:
        wd.stop.set()
    vocals_out /= np.maximum(wsum[0], 1e-8)
    return vocals_out


# ────────────────────────────────────────────── CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="bs-roformer-revive-v3 音轨分离（vocals / other）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    ap.add_argument("-i", "--input", type=Path, required=True, help="输入音频")
    ap.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="输出目录（默认 output/separated/）")
    ap.add_argument("--device", default="cuda", help="cuda（ROCm）/ cpu")
    ap.add_argument("--stems", default="",
                    help="只导出这些轨，逗号分隔，如 vocals（默认全导）")
    ap.add_argument("--no-mix-check", action="store_true", help="不导出 2 轨相加的校验文件")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与耗时预估")
    ap.add_argument("--block-timeout", type=float, default=300.0,
                    help="单块无进度的看门狗阈值秒数，0=关闭（默认 300）")
    ap.add_argument("--no-cache", action="store_true",
                    help="忽略缓存，强制重新推理")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        log(f"✗ 输入不存在: {args.input}")
        return 1

    cfg = load_config()

    audio_cfg = cfg.get("audio", {})
    infer_cfg = cfg.get("inference", {})
    sr = int(audio_cfg.get("sample_rate", 44100))
    chunk = int(audio_cfg.get("chunk_size", 485100))
    num_overlap = int(infer_cfg.get("num_overlap", 2))

    log("=" * 78)
    log(f"bs-roformer-revive-v3 · vocals/other · ROCm")
    log("=" * 78)
    log(f"输入      {args.input}")
    log(f"权重      {cfg['_weights']}")
    log(f"采样率    {sr} Hz   块长 {chunk} ({chunk/sr:.1f}s)   重叠 {num_overlap}")
    log(f"设备      {args.device}")

    out_dir = args.output_dir or (REPO / "output" / "separated")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    ff = find_ff("ffmpeg.exe")
    if ff is None:
        log("✗ 找不到 ffmpeg")
        return 1
    wav = decode_audio(ff, args.input, sr)

    if args.dry_run:
        hop = max(1, chunk // (num_overlap + 1))
        nblocks = max(1, len(range(0, max(1, wav.shape[0] - chunk + 1), hop)))
        log(f"\n--dry-run：{wav.shape[0]/sr:.1f}s 音频 → 约 {nblocks} 块；未实际推理。")
        return 0

    log(f"\n加载模型…")
    t0 = time.time()
    model = build_model(cfg).to(args.device)
    if args.device == "cuda":
        model = model.to(torch.float32)  # ★ ROCm 上 bf16 的 STFT 会掉精度
    log(f"  · 权重已加载（{time.time()-t0:.1f}s）")

    hop = max(1, chunk // (num_overlap + 1))
    nblocks = max(1, len(list(range(0, max(1, wav.shape[0] - chunk + 1), hop))))

    # 缓存
    cache = out_dir / f"{stem}.revive_vocals.npz"
    if cache.exists() and not args.no_cache:
        log(f"\n命中缓存 {cache.name}（--no-cache 可强制重算）")
        vocals = np.load(cache)["vocals"]
    else:
        log(f"\n分离中：{wav.shape[0]/sr:.1f}s → {nblocks} 块"
            + (f"（看门狗 {args.block_timeout:.0f}s/块）" if args.block_timeout > 0 else ""))
        t0 = time.time()
        vocals = separate_vocals(model, wav, sr, args.device, chunk, num_overlap,
                                 block_timeout=args.block_timeout)
        el = time.time() - t0
        log(f"\n✓ 推理完成 {el/60:.1f}min（{wav.shape[0]/sr/el:.1f}x 实时）")
        np.savez(cache, vocals=vocals)

    # instrumental = 原轨 - 人声（残差）
    instrumental = wav - vocals

    written: list[Path] = []
    want = {s.strip() for s in args.stems.split(",") if s.strip()} or {"vocals", "instrumental"}

    if "vocals" in want:
        p = out_dir / f"{stem}.vocals.wav"
        write_wav(p, vocals, sr)
        rms = float(np.sqrt(np.mean(vocals ** 2)))
        log(f"  → {p.name}   RMS {rms:.5f}")
        written.append(p)

    if "instrumental" in want:
        p = out_dir / f"{stem}.instrumental.wav"
        write_wav(p, instrumental, sr)
        rms = float(np.sqrt(np.mean(instrumental ** 2)))
        log(f"  → {p.name}   RMS {rms:.5f}")
        written.append(p)

    if not args.no_mix_check:
        mix = vocals + instrumental
        p = out_dir / f"{stem}.mix_check.wav"
        write_wav(p, mix, sr)
        err = float(np.sqrt(np.mean((mix - wav) ** 2)))
        ref = float(np.sqrt(np.mean(wav ** 2)))
        log(f"  → {p.name}   与原轨 RMS 误差 {err:.5f}（原轨 {ref:.5f}，"
            f"相对 {err/max(ref,1e-9)*100:.2f}%）")

    log(f"\n产物目录 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())