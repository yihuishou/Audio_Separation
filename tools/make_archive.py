# -*- coding: utf-8 -*-
"""把构建产物打成 zip（跨平台统一，CI 与本地共用）。

★ 为什么要单独一个工具：``tools/build_windows.ps1`` 与 ``tools/build_linux.sh``
   原本各自打各自的 zip，**内部结构不一样**：

   - Windows: ``Compress-Archive -Path <目录>\\*`` → zip 里是散文件，**没有顶层目录**
   - Linux  : ``zip -qr ... audio_separation``  → zip 里**有顶层目录**

   同一个 tag 打出来的两个包，解压出来的目录结构却不一样 —— 用户得自己分辨
   哪个是程序、哪个是数据。这里统一成**都带顶层目录**：

       audio_separation/
       ├── audio_separation.exe      (Windows)
       ├── audio_separation          (Linux)
       └── _internal/

   解压出来就是一个自洽的目录，拷走即用。

★ 为什么不用 ``zip`` 命令：Windows 上没有它；``tar`` 两个平台都有但生成的是
   tar.gz。Python 的 ``zipfile`` 两个平台都有，行为一致，不需要额外装东西。

★ 可执行位：Linux 产物的 ``audio_separation`` 必须有 +x，否则解压出来跑不了。
   ``zipfile`` 存 Unix 权限靠 ``external_attr``，写的时候要手动补上。
"""

from __future__ import annotations

import argparse
import os
import platform
import stat
import sys
import time
import zipfile
from pathlib import Path

from console_utf8 import force_utf8

force_utf8()

REPO = Path(__file__).resolve().parent.parent

#: 压缩包里顶层目录的名字（与 PyInstaller 的 COLLECT name 一致）
TOP_DIR = "audio_separation"

#: Linux 上必须保留 +x 的文件
EXECUTABLE_NAMES = {"audio_separation", "audio_separation.exe"}

#: 机器架构名 → 通用架构名
_ARCH_ALIASES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "x64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "i386": "x86",
    "i686": "x86",
}


def platform_tag() -> str:
    """当前平台标签，如 ``linux-x86_64`` / ``windows-x86_64``。

    ★ 同一个 tag 会在两个平台各打一个包，文件名必须能区分，
    否则上传 Release 时后一个会顶掉前一个。
    """
    system = platform.system().lower()
    if system == "windows":
        name = "windows"
    elif system == "darwin":
        name = "macos"
    elif system == "linux":
        name = "linux"
    else:
        name = system or "unknown"
    machine = platform.machine().lower()
    arch = _ARCH_ALIASES.get(machine, machine or "unknown")
    return f"{name}-{arch}"


def _is_executable(path: Path) -> bool:
    """判断文件是否需要保留可执行位。

    两个来源：源文件的权限位（Linux 构建时 PyInstaller 收进来的 ffmpeg 等
    可能已经带 +x），以及文件名（主程序本身）。
    """
    if path.name in EXECUTABLE_NAMES:
        return True
    try:
        return bool(path.stat().st_mode & stat.S_IXUSR)
    except OSError:
        return False


def _zip_timestamp(path: Path) -> tuple[int, int, int, int, int, int]:
    """取文件的修改时间，转成 ZipInfo 要的 6 元组 (年, 月, 日, 时, 分, 秒)。

    ``st_mtime`` 是浮点秒数，不是元组——直接切片会报
    ``TypeError: 'float' object is not subscriptable``。
    zip 格式最早只支持 1980 年，早于那年的时间要抬到 1980-01-01。
    """
    tm = time.localtime(path.stat().st_mtime)
    year = max(tm.tm_year, 1980)
    return (year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)


def _make_zipinfo(path: Path, arcname: str) -> zipfile.ZipInfo:
    """构造保留可执行位的 ZipInfo。

    ``ZipFile.write()`` 不会保留 Unix 权限位 —— 它把 external_attr 设成 0。
    解压出来的 Linux 程序因此会丢掉 +x，直接 Permission denied。
    这里显式把源文件的 mode 写进去。
    """
    info = zipfile.ZipInfo(arcname, date_time=_zip_timestamp(path))
    info.compress_type = zipfile.ZIP_DEFLATED

    if _is_executable(path):
        # 0755：任何人可执行，只有属主可写
        mode = 0o755
    else:
        mode = 0o644
    # external_attr 高 16 位存 Unix mode，低位放 MS-DOS 属性
    info.external_attr = (mode & 0xFFFF) << 16

    if path.is_dir():
        info.external_attr |= 0x10  # MS-DOS 目录标志
    return info


def build_zip(app_dir: Path, out_zip: Path) -> Path:
    """把 app_dir 打成 out_zip，内部统一套一层 TOP_DIR。"""
    if not app_dir.is_dir():
        raise SystemExit(f"找不到构建产物目录：{app_dir}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(app_dir.rglob("*")):
            if path.is_dir():
                continue
            arcname = f"{TOP_DIR}/{path.relative_to(app_dir).as_posix()}"
            zf.writestr(_make_zipinfo(path, arcname), path.read_bytes())
            count += 1

    size_mb = out_zip.stat().st_size / 1024 / 1024
    print(f"[make_archive] 已生成: {out_zip}")
    print(f"[make_archive] 文件数: {count}")
    print(f"[make_archive] 体积:   {size_mb:.1f} MB")
    print(f"[make_archive] 顶层目录: {TOP_DIR}/")
    return out_zip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把构建产物打成 zip")
    parser.add_argument(
        "--app-dir",
        default="dist/audio_separation",
        help="构建产物目录（默认 dist/audio_separation）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出的 zip 路径（默认 dist/audio-separation-<平台>.zip）",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="版本号，拼进输出文件名（如 1.2.3 → audio-separation-1.2.3-<平台>.zip）",
    )
    args = parser.parse_args(argv)

    app_dir = (REPO / args.app_dir).resolve()
    if args.out:
        out_zip = Path(args.out).resolve()
    else:
        parts = ["audio-separation"]
        if args.version:
            parts.append(args.version)
        parts.append(platform_tag())
        out_zip = (REPO / "dist" / ("-".join(parts) + ".zip")).resolve()

    build_zip(app_dir, out_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
