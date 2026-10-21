# -*- coding: utf-8 -*-
"""把版本号写进源码（CI 按 tag 版本编译时用）。

★ 为什么要单独一个工具：版本号在仓库里有**两处**硬编码：

    src/audio_separation/__init__.py:15   __version__ = "0.1.0"
    pyproject.toml:10                     version = "0.1.0"

``--version`` 读的是前者，``pip install .`` 读的是后者。
CI 拿 tag（如 ``v1.2.3``）编译时必须两处一起改，否则会出现
「包里显示 0.1.0，但安装元数据说 1.2.3」这种对不上的情况。

## 用法

    python tools/set_version.py 1.2.3
    python tools/set_version.py v1.2.3      # 前缀 v 自动去掉
    python tools/set_version.py 1.2.3 --check   # 只校验不写

★ 幂等：同样的版本号跑两次，文件内容不变（不会叠加改动）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from console_utf8 import force_utf8

force_utf8()

REPO = Path(__file__).resolve().parent.parent

#: 版本号格式：数字.数字.数字，可带 -rc.1 / +build.5 这样的后缀
VERSION_RE = re.compile(
    r"^\d+\.\d+\.\d+(?:[-.]?(?:a|b|rc|alpha|beta|dev|post)\.?\d*)?(?:\+[0-9A-Za-z.-]+)?$"
)

#: __init__.py 里的版本行
INIT_PATTERN = re.compile(r'^(__version__\s*=\s*")([^"]+)(".*)$', re.MULTILINE)

#: pyproject.toml 里的版本行（只改 [project] 段那行，避免动到别的 version 字段）
PYPROJECT_PATTERN = re.compile(r'^(version\s*=\s*")([^"]+)(".*)$', re.MULTILINE)


def normalize(raw: str) -> str:
    """把 ``v1.2.3`` 规范成 ``1.2.3``，并校验格式。"""
    version = raw.strip()
    if version.startswith("v"):
        version = version[1:]
    if not VERSION_RE.match(version):
        raise ValueError(
            f"版本号格式不合法：{raw!r}\n"
            f"期望形如 1.2.3 / 1.2.3rc1 / 1.2.3-dev.1"
        )
    return version


def _rewrite(path: Path, pattern: re.Pattern, version: str) -> bool:
    """把文件里匹配到的版本号改成 version，返回是否发生了改动。"""
    # 读写都指定 newline：读时把 CRLF 统一成 LF，写时原样输出 LF。
    # ★ 不能用 write_text —— 它在 Windows 上会把整个文件转成 CRLF，
    #   改个版本号却让整份文件的行尾都变了。
    with path.open("r", encoding="utf-8", newline=None) as f:
        text = f.read()
    new_text, count = pattern.subn(rf"\g<1>{version}\g<3>", text)
    if count == 0:
        raise RuntimeError(f"在 {path} 里没找到版本号那一行，文件结构可能变了")
    if new_text == text:
        return False
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)
    return True


def current_versions() -> dict[str, str]:
    """读出两处当前的版本号，用于 --check。"""
    init_text = (REPO / "src" / "audio_separation" / "__init__.py").read_text(
        encoding="utf-8"
    )
    proj_text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    init_match = INIT_PATTERN.search(init_text)
    proj_match = PYPROJECT_PATTERN.search(proj_text)
    if not init_match or not proj_match:
        raise RuntimeError("读版本号失败：文件结构与预期不符")
    return {
        "src/audio_separation/__init__.py": init_match.group(2),
        "pyproject.toml": proj_match.group(2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把版本号写进源码")
    parser.add_argument("version", nargs="?", help="版本号，如 1.2.3 或 v1.2.3")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验、不写文件。给出版本号则校验两处是否都等于它；不给则只校验两处是否彼此一致",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="只打印当前两处的版本号，不写文件",
    )
    args = parser.parse_args(argv)

    current = current_versions()

    if args.show:
        for path, value in current.items():
            print(f"[set_version] {path} = {value}")
        values = set(current.values())
        if len(values) > 1:
            print(
                f"[set_version] ★ 两处版本号不一致：{sorted(values)}",
                file=sys.stderr,
            )
            return 1
        return 0

    if not args.version:
        if args.check:
            values = set(current.values())
            if len(values) > 1:
                print(f"[set_version] ★ 两处版本号不一致：{sorted(values)}", file=sys.stderr)
                for path, value in current.items():
                    print(f"           {path} = {value}", file=sys.stderr)
                return 1
            print(f"[set_version] 两处版本号一致：{current[next(iter(current))]}")
            return 0
        parser.error("必须给出版本号（或用 --show 只查看，或用 --check 只校验一致性）")

    try:
        version = normalize(args.version)
    except ValueError as exc:
        print(f"[set_version] {exc}", file=sys.stderr)
        return 2

    if args.check:
        mismatched = {k: v for k, v in current.items() if v != version}
        if mismatched:
            print(f"[set_version] 版本号不一致，期望 {version}：", file=sys.stderr)
            for path, value in mismatched.items():
                print(f"           {path} = {value}", file=sys.stderr)
            return 1
        print(f"[set_version] 版本号一致：{version}")
        return 0

    changed = False
    for path, pattern in (
        (REPO / "src" / "audio_separation" / "__init__.py", INIT_PATTERN),
        (REPO / "pyproject.toml", PYPROJECT_PATTERN),
    ):
        rel = path.relative_to(REPO).as_posix()
        if _rewrite(path, pattern, version):
            changed = True
            print(f"[set_version] {rel}: {current[rel]} -> {version}")
        else:
            print(f"[set_version] {rel}: 已是 {version}，无需改动")

    print(f"[set_version] 完成{'（有改动）' if changed else '（无改动）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
