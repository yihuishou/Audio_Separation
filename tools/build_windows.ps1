<#
.SYNOPSIS
    构建 Windows 可执行程序（PLAN §2.13 / M7）。

.DESCRIPTION
    产物是 onedir 目录 dist\audio_separation\，里面只有「程序」——
    torch / numpy / scipy / librosa / BS-RoFormer 由**目标系统**自己装
    （见 README 分发章节）。

    models\ 与 ffmpeg\ 不进包（合计 1.5 GB），构建完按提示手动复制到 exe 旁边。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1
    powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -Zip
    powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -Version 1.2.3 -Zip
#>
[CmdletBinding()]
param(
    # 构建完顺手打个 zip（只含程序，不含 models/ffmpeg）
    [switch]$Zip,
    # 版本号：写进源码后编译。CI 按 tag 版本编译时用这个。
    [string]$Version
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo

try {
    $python = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $python) {
        throw "找不到 python，请先安装 Python 3.11+ 并加入 PATH"
    }
    Write-Host "[1/5] 使用 Python: $($python.Source)" -ForegroundColor Cyan

    & $python.Source -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "     PyInstaller 未安装，正在安装..." -ForegroundColor Yellow
        & $python.Source -m pip install --upgrade pyinstaller
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller 安装失败" }
    }
    $pyiVer = & $python.Source -m PyInstaller --version
    Write-Host "     $pyiVer" -ForegroundColor DarkGray

    if ($Version) {
        Write-Host "[2/5] 写入版本号: $Version" -ForegroundColor Cyan
        & $python.Source tools\set_version.py $Version
        if ($LASTEXITCODE -ne 0) { throw "版本号写入失败" }
    } else {
        Write-Host "[2/5] 沿用源码里的版本号（未指定 -Version）" -ForegroundColor Cyan
        & $python.Source tools\set_version.py --show
        if ($LASTEXITCODE -ne 0) { throw "读取版本号失败" }
    }

    Write-Host "[3/5] 清理旧的 build\ 与 dist\" -ForegroundColor Cyan
    foreach ($dir in @("build", "dist")) {
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    }

    Write-Host "[4/5] 运行 PyInstaller（首次约需 3-10 分钟）..." -ForegroundColor Cyan
    & $python.Source -m PyInstaller packaging\audio_separation.spec --noconfirm --clean
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败" }

    $appDir = Join-Path $repo "dist\audio_separation"
    $exe = Join-Path $appDir "audio_separation.exe"
    if (-not (Test-Path $exe)) { throw "构建结束但没找到产物：$exe" }

    $sizeMB = [math]::Round(((Get-ChildItem $appDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
    Write-Host "[5/5] 构建完成" -ForegroundColor Green
    Write-Host "     产物: $exe" -ForegroundColor Green
    Write-Host "     体积: $sizeMB MB" -ForegroundColor Green

    Write-Host ""
    Write-Host "接下来需要你手动准备两样东西（不进包）：" -ForegroundColor Yellow
    Write-Host "  1. models\   （约 1.2 GB）→ 复制到 $appDir\models" -ForegroundColor Yellow
    Write-Host "  2. ffmpeg\   （约 315 MB）→ 复制到 $appDir\ffmpeg" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "目标系统还需要自己装推理栈（README「分发」章节有完整命令）：" -ForegroundColor Yellow
    Write-Host "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2" -ForegroundColor Yellow
    Write-Host "  pip install BS-RoFormer==1.2.4" -ForegroundColor Yellow
    Write-Host "  ★ 顺序不能反：BS-RoFormer 声明 torch>=2.0，torch 没先装的话" -ForegroundColor Yellow
    Write-Host "    pip 会从 PyPI 拉默认的 CUDA 构建，在 AMD 机器上装出用不了的 torch。" -ForegroundColor Yellow

    if ($Zip) {
        Write-Host ""
        Write-Host "     正在打包 zip（不含 models/ffmpeg）..." -ForegroundColor Cyan
        # ★ 用 tools\make_archive.py 而不是 Compress-Archive：
        #   Compress-Archive 打出来的 zip 里是散文件（没有顶层目录），
        #   而 Linux 侧是带顶层目录的 —— 同一个 tag 的两个包结构不一致。
        #   make_archive.py 两个平台行为一致，zip 内统一是 audio_separation/。
        $zipArgs = @("--app-dir", "dist\audio_separation")
        if ($Version) { $zipArgs += @("--version", $Version) }
        & $python.Source tools\make_archive.py @zipArgs
        if ($LASTEXITCODE -ne 0) { throw "打包 zip 失败" }
    }
}
finally {
    Pop-Location
}
