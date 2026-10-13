# -*- coding: utf-8 -*-
"""生成 1 分钟测试切片（PLAN §1.2）。

从 `test_audio/` 下的全曲 wav 中提取 60 秒切片到 `test_audio/clips/`，
用于快速验证分离质量而无需等待完整 15 分钟推理。

为什么是 60 秒而不是 10 秒：
模型推理的 chunk_size = 485100 采样（44100 Hz 下约 11.0 秒），
hop = 485100/3 = 161700 采样（约 3.667 秒）。
60 秒切片产生 14 个分块和 13 个内部重叠边界，充分覆盖 num_overlap 与
窗口和归一化逻辑。10 秒只有一个块，完全无法测试重叠合并。

为什么保留原始 wav：
切片可随时重新生成，原始素材是唯一真源。原始 wav 永远不被修改。

Unicode 路径处理：
切片文件名从输入 stem 派生，保留日文字符和全角括号『』。
这既是功能需求也是测试点：验证 subprocess、CreateProcessW、ffmpeg 和
文件系统都能正确处理 Unicode 路径。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 源文件（PLAN §1.2 中列出的确切文件名）
SOURCE_FILES = [
    "1.第1章『耳なめ手コキオナサポ』.wav",
    "2.第2章『乳首責め手コキオナサポ』.wav",
]


def resolve_ffmpeg() -> tuple[Path, Path]:
    """定位 ffmpeg 和 ffprobe。

    唯一真源是 `audio_separation.paths`（PLAN §2.10）：仓库里不允许出现第二份
    路径解析逻辑，否则两份实现必然各自漂移。没安装成包时把 `src/` 加进
    `sys.path` 再试一次；仍拿不到才回退到本地路径计算。

    不搜索系统 PATH —— 只用仓库内捆绑的二进制（PLAN §0 决策）。
    """
    for attempt in (0, 1):
        try:
            from audio_separation.paths import ffmpeg_bin, ffprobe_bin
        except ImportError:
            if attempt == 0:
                # 未 `pip install -e .`：把 src/ 挂到 sys.path 再试一次
                src_dir = Path(__file__).resolve().parent.parent / "src"
                if src_dir.is_dir() and str(src_dir) not in sys.path:
                    sys.path.insert(0, str(src_dir))
                continue
            break
        return ffmpeg_bin(), ffprobe_bin()

    # 回退：相对于此脚本的本地路径计算
    repo_root = Path(__file__).resolve().parent.parent
    exe = ".exe" if sys.platform == "win32" else ""
    bin_dir = repo_root / "ffmpeg" / "bin"
    return bin_dir / f"ffmpeg{exe}", bin_dir / f"ffprobe{exe}"


def ffprobe_duration(ffprobe: Path, file_path: Path) -> float | None:
    """使用 ffprobe 查询文件时长（秒）。"""
    try:
        result = subprocess.run(
            [
                str(ffprobe),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError):
        return None


def ffprobe_file_info(ffprobe: Path, file_path: Path) -> dict | None:
    """查询文件的完整音频信息。"""
    try:
        result = subprocess.run(
            [
                str(ffprobe),
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name,sample_rate,channels",
                "-show_entries", "format=duration,size",
                "-of", "json",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def make_clip(ffmpeg: Path, src: Path, dst: Path, t0: float, duration: float) -> bool:
    """使用 ffmpeg 从源生成一个切片。

    ffmpeg 命令：
    ffmpeg -nostdin -y -ss <t0> -i <src> -t 60 -c:a pcm_s24le -ar 44100 -ac 2 <dst>

    ★ -ss 放在 -i 之前（输入侧快速定位，秒级）。放在 -i 之后会先解码
    完整文件，在 30 分钟素材上会明显变慢。

    输出参数与源完全匹配（pcm_s24le/44100/2ch）——没有重编码降级。
    """
    cmd = [
        str(ffmpeg),
        "-nostdin",
        "-y",
        "-ss", str(t0),
        "-i", str(src),
        "-t", str(duration),
        "-c:a", "pcm_s24le",
        "-ar", "44100",
        "-ac", "2",
        str(dst),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ ffmpeg 失败（退出码 {e.returncode}）", file=sys.stderr)
        if e.stderr:
            print(f"    stderr: {e.stderr.strip()}", file=sys.stderr)
        return False


def clip_filename(src_name: str, t0: int) -> str:
    """构造切片文件名：`<原stem>_t{偏移}s.wav`。"""
    stem = Path(src_name).stem
    return f"{stem}_t{t0}s.wav"


def format_size(size_bytes: int) -> str:
    """将字节格式化为人类可读的大小。"""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def main():
    parser = argparse.ArgumentParser(
        description="从全曲生成 1 分钟测试切片（PLAN §1.2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src-dir",
        default=None,
        help="源音频目录（默认：test_audio/）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="切片输出目录（默认：test_audio/clips/）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="切片时长（秒，默认：30.0）",
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default="240,600",
        help="切片偏移（秒，逗号分隔，默认：240,600）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新编码，即使切片已存在",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅报告已存在/缺失的切片，不生成",
    )

    args = parser.parse_args()

    # 解析偏移
    try:
        offsets = [int(o) for o in args.offsets.split(",")]
    except ValueError:
        print(f"错误：无效的偏移值：{args.offsets}", file=sys.stderr)
        sys.exit(1)

    # 解析路径
    ffmpeg_bin_path, ffprobe_bin_path = resolve_ffmpeg()
    repo_root = Path(__file__).resolve().parent.parent

    if args.src_dir:
        src_dir = Path(args.src_dir)
    else:
        src_dir = repo_root / "test_audio"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = src_dir / "clips"

    # 验证 ffmpeg 二进制文件
    if not ffmpeg_bin_path.exists():
        print(f"错误：ffmpeg 不存在：{ffmpeg_bin_path}", file=sys.stderr)
        sys.exit(1)
    if not ffprobe_bin_path.exists():
        print(f"错误：ffprobe 不存在：{ffprobe_bin_path}", file=sys.stderr)
        sys.exit(1)

    # 验证源目录
    if not src_dir.exists():
        print(f"错误：源目录不存在：{src_dir}", file=sys.stderr)
        print(f"期望的文件：{', '.join(SOURCE_FILES)}", file=sys.stderr)
        sys.exit(1)

    # 验证源文件
    source_paths = []
    for name in SOURCE_FILES:
        path = src_dir / name
        if not path.exists():
            print(f"错误：源文件不存在：{path}", file=sys.stderr)
            sys.exit(1)
        source_paths.append(path)

    # 创建输出目录
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集切片计划
    plan = []
    for src in source_paths:
        for t0 in offsets:
            dst = out_dir / clip_filename(src.name, t0)
            plan.append({
                "src": src,
                "dst": dst,
                "t0": t0,
            })

    # 执行
    print(f"切片生成器（PLAN §1.2）")
    print(f"  源目录：{src_dir}")
    print(f"  输出目录：{out_dir}")
    print(f"  偏移：{offsets}")
    print(f"  时长：{args.duration} 秒")
    print(f"  ffmpeg：{ffmpeg_bin_path.name}")
    print(f"  计划：{len(plan)} 个切片")
    print()

    created = 0
    skipped = 0
    would_create = 0
    errors = 0

    for item in plan:
        src = item["src"]
        dst = item["dst"]
        t0 = item["t0"]

        print(f"处理：{dst.name}")

        if args.check:
            # 仅检查模式
            if not dst.exists():
                print(f"  缺失")
                would_create += 1
            else:
                dur = ffprobe_duration(ffprobe_bin_path, dst)
                if dur is not None and abs(dur - args.duration) <= 0.05:
                    print(f"  已存在（{dur:.1f}s）— 正确")
                else:
                    print(f"  已存在（{dur:.1f}s）— 长度错误，需要重新生成")
                    would_create += 1
            continue

        # 幂等性：如果已存在且长度正确，则跳过
        if not args.force and dst.exists():
            dur = ffprobe_duration(ffprobe_bin_path, dst)
            if dur is not None and abs(dur - args.duration) <= 0.05:
                size = dst.stat().st_size
                print(f"  已存在（{dur:.1f}s, {format_size(size)}）— 跳过")
                skipped += 1
                continue

        # 生成切片
        if not make_clip(ffmpeg_bin_path, src, dst, float(t0), args.duration):
            errors += 1
            continue

        # 验证生成的文件
        dur = ffprobe_duration(ffprobe_bin_path, dst)
        size = dst.stat().st_size
        if dur is not None:
            print(f"  已生成（{dur:.1f}s, {format_size(size)}）")
        else:
            print(f"  已生成（{format_size(size)}）")
        created += 1

    # 摘要
    print()
    print("摘要：")
    print(f"  已生成：{created}")
    print(f"  已跳过：{skipped}")
    if would_create:
        print(f"  待生成：{would_create}")
    print(f"  错误：{errors}")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
