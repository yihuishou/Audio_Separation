# -*- coding: utf-8 -*-
"""校验打包产物：存在、能跑、版本号对得上。

CI 与本地共用。之所以单独写这个脚本而不是把逻辑塞进工作流里：
工作流里的多行内联 Python 在 Windows runner 上转义极易出错
（PowerShell 与 bash 的引号规则不同），而这一步是"发出去之前最后一道闸"，
出错时必须一眼看出是哪一环坏了。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def find_executable(app_dir: Path) -> Path:
    """在产物目录里找到主程序。"""
    for name in ("audio_separation.exe", "audio_separation"):
        candidate = app_dir / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"产物目录里找不到主程序：{app_dir}")


def read_version(exe: Path, timeout: int = 120) -> str:
    """跑 ``--version`` 拿回版本号。

    ★ Windows 上产物是 GUI 子系统（console=False），但**父进程用管道拉起时
      stdout 依然可用** —— PyInstaller 只影响 exe 自己的控制台分配方式。
    """
    proc = subprocess.run(
        [str(exe), "--version"],
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"运行 {exe.name} --version 失败，退出码 {proc.returncode}\n"
            f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
        )
    return proc.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验打包产物")
    parser.add_argument(
        "--app-dir",
        default="dist/audio_separation",
        help="构建产物目录（默认 dist/audio_separation）",
    )
    parser.add_argument(
        "--expect",
        default=None,
        help="期望的版本号；不给就只检查能不能跑起来",
    )
    args = parser.parse_args(argv)

    app_dir = (REPO / args.app_dir).resolve()
    if not app_dir.is_dir():
        raise SystemExit(f"找不到构建产物目录：{app_dir}")

    exe = find_executable(app_dir)
    print(f"[verify] 主程序: {exe}")

    if sys.platform != "win32" and not exe.stat().st_mode & 0o111:
        raise SystemExit(f"主程序没有可执行位：{exe}")

    actual = read_version(exe)
    print(f"[verify] 产物版本号: {actual}")

    if args.expect is None:
        print("[verify] ✓ 产物能正常启动")
        return 0

    if actual != args.expect:
        raise SystemExit(f"版本号不符：期望 {args.expect}，实际 {actual}")
    print(f"[verify] ✓ 版本号一致：{actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
