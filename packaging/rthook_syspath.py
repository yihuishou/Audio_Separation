# -*- coding: utf-8 -*-
"""PyInstaller runtime hook：把**目标系统**的 site-packages 加回 `sys.path`。

## 为什么需要这个文件

PyInstaller 打包后，`sys.path` 只剩三项，且全在 bundle 内部：

    <dist>/app/_internal/base_library.zip
    <dist>/app/_internal/python3.12/lib-dynload
    <dist>/app/_internal

目标系统 `pip install torch` 装的那份 torch **根本不在搜索路径里**，
表现就是 `ModuleNotFoundError: No module named 'torch'`。

（这正是 PyInstaller 官方 discussion #8464 那个问题的根因。）

runtime hook 是 bootloader 之后、用户代码之前唯一能改 `sys.path` 的时机。

## 探测链（按可靠性从高到低）

1. 环境变量 ``AUDIO_SEPARATION_SITE_PACKAGES``（``os.pathsep`` 分隔）—— 最可靠
2. ``site.getusersitepackages()``
3. Windows 注册表 ``HKCU/HKLM\\SOFTWARE\\Python\\PythonCore\\*\\InstallPath``
4. POSIX：扫 ``PATH`` 里的 python3，逐个问它 ``sysconfig.get_path('purelib')``
5. 常见安装路径 glob

## 为什么要追加到 sys.path **末尾**

bundle 内的模块优先，系统的只在 bundle 里没有时生效。

反过来（插到开头）会出现：bundle 里那份 numpy 先被 import，
而系统 torch 是按**它自己那份** numpy 的版本编译的 —— 版本对不上会直接崩。
"""
from __future__ import annotations

import glob
import os
import site
import sys

#: 显式指定系统 site-packages 的环境变量（多个用 os.pathsep 分隔）
ENV_VAR = "AUDIO_SEPARATION_SITE_PACKAGES"


def _windows_registry_candidates() -> list[str]:
    """从注册表找已安装的 Python。

    键路径：``HKCU/HKLM\\SOFTWARE\\Python\\PythonCore\\<版本>\\InstallPath``，
    site-packages 在 ``<InstallPath>\\Lib\\site-packages``。
    """
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    out: list[str] = []
    for hkey in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            core = winreg.OpenKey(hkey, r"SOFTWARE\Python\PythonCore")
        except OSError:
            continue
        idx = 0
        while True:
            try:
                ver = winreg.EnumKey(core, idx)
            except OSError:
                break
            idx += 1
            try:
                install_key = winreg.OpenKey(core, ver + r"\InstallPath")
                root, _ = winreg.QueryValueEx(install_key, "")
            except OSError:
                continue
            if root:
                out.append(os.path.join(root, "Lib", "site-packages"))
    return out


def _posix_candidates() -> list[str]:
    """Linux/macOS：扫 PATH 里的 python3，逐个问它 purelib 在哪。"""
    if sys.platform == "win32":
        return []
    out: list[str] = []
    for exe in os.environ.get("PATH", "").split(os.pathsep):
        exe = exe.strip()
        if not exe or not os.path.isfile(exe):
            continue
        name = os.path.basename(exe).lower()
        if not name.startswith("python"):
            continue
        try:
            import subprocess

            res = subprocess.run(
                [exe, "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            continue
        if res.returncode == 0:
            path = res.stdout.strip()
            if path:
                out.append(path)
    return out


def _glob_candidates() -> list[str]:
    pats = [
        r"C:\Python*\Lib\site-packages",
        r"C:\Program Files\Python\Python3*\Lib\site-packages",
        r"C:\Program Files (x86)\Python\Python3*\Lib\site-packages",
        r"C:\Users\*\AppData\Local\Programs\Python\Python3*\Lib\site-packages",
        r"/usr/lib/python3*/site-packages",
        r"/usr/local/lib/python3*/site-packages",
        r"/opt/homebrew/lib/python3*/site-packages",
    ]
    out: list[str] = []
    for pat in pats:
        out.extend(glob.glob(pat))
    return out


def find_system_site_packages() -> list[str]:
    """探测目标系统上已安装的 site-packages，返回存在的目录列表。"""
    candidates: list[str] = []

    env = os.environ.get(ENV_VAR)
    if env:
        candidates.extend(p for p in env.split(os.pathsep) if p)

    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    candidates.extend(_windows_registry_candidates())
    candidates.extend(_posix_candidates())
    candidates.extend(_glob_candidates())

    seen: set[str] = set()
    out: list[str] = []
    for path in candidates:
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isdir(norm):
            out.append(norm)
    return out


def apply() -> None:
    """把探测到的系统 site-packages 追加到 sys.path 末尾。

    找不到就什么都不做 —— 让后续 import 报原始的 ModuleNotFoundError，
    比在这里抛一个更让人困惑的异常要好。
    """
    for path in find_system_site_packages():
        if path not in sys.path:
            sys.path.append(path)


apply()
