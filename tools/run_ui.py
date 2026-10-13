#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音声分离工具 GUI 启动入口。

用法: python tools/run_ui.py
"""
import sys
from pathlib import Path

# 支持直接运行(python tools/run_ui.py),无需安装
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_separation.ui.app import main

if __name__ == "__main__":
    sys.exit(main())
