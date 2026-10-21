#!/usr/bin/env bash
# 构建 Linux 可执行程序（PLAN §2.13 / M7）。
#
# 产物是 onedir 目录 dist/audio_separation/，里面只有「程序」——
# torch / numpy / scipy / librosa / BS-RoFormer 由目标系统自己装
# （见 README 分发章节）。
#
# ★ PyInstaller 不能跨平台编译：Linux 包必须在 Linux 上构建。
#
# 用法：
#   bash tools/build_linux.sh
#   bash tools/build_linux.sh --zip
#   bash tools/build_linux.sh --version 1.2.3 --zip

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MAKE_ZIP=0
VERSION=""
while [ $# -gt 0 ]; do
    case "$1" in
        --zip) MAKE_ZIP=1; shift ;;
        --version)
            [ $# -ge 2 ] || { echo "--version 后面要跟版本号" >&2; exit 2; }
            VERSION="$2"; shift 2 ;;
        --version=*)
            VERSION="${1#--version=}"; shift ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

PYTHON="${PYTHON:-python3}"

echo "[1/5] 使用 Python: $($PYTHON --version 2>&1)"

if ! "$PYTHON" -c "import PyInstaller" 2>/dev/null; then
    echo "     PyInstaller 未安装，正在安装..." >&2
    "$PYTHON" -m pip install --upgrade pyinstaller
fi
"$PYTHON" -m PyInstaller --version | sed 's/^/     /' >&2

if [ -n "$VERSION" ]; then
    echo "[2/5] 写入版本号: $VERSION"
    "$PYTHON" tools/set_version.py "$VERSION"
else
    echo "[2/5] 沿用源码里的版本号（未指定 --version）"
    "$PYTHON" tools/set_version.py --show
fi

echo "[3/5] 清理旧的 build/ 与 dist/"
rm -rf build dist

echo "[4/5] 运行 PyInstaller（首次约需 3-10 分钟）..."
"$PYTHON" -m PyInstaller packaging/audio_separation.spec --noconfirm --clean

APP_DIR="$REPO/dist/audio_separation"
if [ ! -x "$APP_DIR/audio_separation" ]; then
    echo "构建结束但没找到产物：$APP_DIR/audio_separation" >&2
    exit 1
fi

# ★ PyInstaller 的 --add-binary 不保留权限位，收集进来的 ffmpeg/ffprobe
#   会丢掉 +x，运行时直接 Permission denied。PyInstaller 官方 issue #3926
#   承认它不保留文件属性。src/audio_separation/core/audio_io.py 里已有
#   运行时 chmod 兜底，这里再补一次，保证用户手动放置的 ffmpeg 也能用。
echo "     修复可执行权限位..."
chmod +x "$APP_DIR/audio_separation"
if [ -d "$APP_DIR/ffmpeg/bin" ]; then
    chmod +x "$APP_DIR"/ffmpeg/bin/* 2>/dev/null || true
fi

SIZE_MB=$(du -sm "$APP_DIR" | cut -f1)

echo "[5/5] 构建完成"
echo "     产物: $APP_DIR/audio_separation"
echo "     体积: ${SIZE_MB} MB"

cat <<'EOF'

接下来需要你手动准备两样东西（不进包）：
  1. models/   （约 1.2 GB）→ 复制到 dist/audio_separation/models
  2. ffmpeg/   （约 315 MB）→ 复制到 dist/audio_separation/ffmpeg
     ★ 必须是 Linux 版二进制（ffmpeg / ffprobe，无 .exe 后缀），
       Windows 版在 Linux 上跑不了。

目标系统还需要自己装推理栈（README「分发」章节有完整命令）：
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
  pip install BS-RoFormer==1.2.4
  ★ 顺序不能反：BS-RoFormer 声明 torch>=2.0，torch 没先装的话 pip 会从
    PyPI 拉默认的 CUDA 构建，在 AMD 机器上装出用不了的 torch。
  （numpy / scipy / librosa 都在 BS-RoFormer 这条依赖链里，不用单独装。）

★ BS-RoFormer 必须装：PoPE_pytorch 里的 @triton.jit 需要读源码，
  打包后 .pyc 里读不到 → 冻结态一启动就崩（与 CUDA 无关）。

★ ROCm 驱动是**系统级**的（amdgpu 内核模块），无法随包分发，
  目标机器必须先装好对应发行版的 ROCm 驱动。
EOF

if [ "$MAKE_ZIP" = "1" ]; then
    echo
    echo "     正在打包 zip（不含 models/ffmpeg）..."
    # ★ 用 tools/make_archive.py 而不是 zip 命令：Windows 上没有 zip 命令，
    #   且两边 zip 内部结构要一致（统一带顶层 audio_separation/ 目录）。
    #   它还负责保留可执行位（zipfile 默认会丢掉 Unix 权限位）。
    if [ -n "$VERSION" ]; then
        "$PYTHON" tools/make_archive.py --app-dir dist/audio_separation --version "$VERSION"
    else
        "$PYTHON" tools/make_archive.py --app-dir dist/audio_separation
    fi
fi
