"""audio_separation 模块入口

用法: python -m audio_separation
"""

import importlib.util
import sys


def _try_import_app():
    """尝试导入 ui.app 模块，如果不存在则返回 None。"""
    try:
        # 尝试使用 importlib 导入 ui.app，如果不存在则抛出 ModuleNotFoundError
        spec = importlib.util.find_spec("audio_separation.ui.app")
        if spec is None:
            return None
        return importlib.util.module_from_spec(spec)
    except (ImportError, ModuleNotFoundError):
        return None


def main():
    """主入口点。"""
    app_module = _try_import_app()
    if app_module is not None and hasattr(app_module, "main"):
        app_module.main()
    else:
        print("audio_separation - BS-RoFormer 音声分离图形化工具")
        print(f"版本: {__import__('audio_separation').__version__}")
        print()
        print("UI 模块尚未实现。请等待后续版本或运行:")
        print("  python -m audio_separation.paths  # 查看路径解析状态")
        print("  python tools/bs_roformer_revive_separate.py  # 使用 CLI 参考脚本")


if __name__ == "__main__":
    main()