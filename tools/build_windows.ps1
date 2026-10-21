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
#>
[CmdletBinding()]
param(
    # 构建完顺手打个 zip（只含程序，不含 models/ffmpeg）
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo

try {
    $python = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $python) {
        throw "找不到 python，请先安装 Python 3.11+ 并加入 PATH"
    }
    Write-Host "[1/4] 使用 Python: $($python.Source)" -ForegroundColor Cyan

    & $python.Source -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "     PyInstaller 未安装，正在安装..." -ForegroundColor Yellow
        & $python.Source -m pip install --upgrade pyinstaller
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller 安装失败" }
    }
    $pyiVer = & $python.Source -m PyInstaller --version
    Write-Host "     $pyiVer" -ForegroundColor DarkGray

    Write-Host "[2/4] 清理旧的 build\ 与 dist\" -ForegroundColor Cyan
    foreach ($dir in @("build", "dist")) {
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    }

    Write-Host "[3/4] 运行 PyInstaller（首次约需 3-10 分钟）..." -ForegroundColor Cyan
    & $python.Source -m PyInstaller packaging\audio_separation.spec --noconfirm --clean
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败" }

    $appDir = Join-Path $repo "dist\audio_separation"
    $exe = Join-Path $appDir "audio_separation.exe"
    if (-not (Test-Path $exe)) { throw "构建结束但没找到产物：$exe" }

    $sizeMB = [math]::Round(((Get-ChildItem $appDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
    Write-Host "[4/4] 构建完成" -ForegroundColor Green
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
        $zip = Join-Path $repo "dist\audio_separation-program-only.zip"
        Write-Host ""
        Write-Host "     正在打包 zip（不含 models/ffmpeg）..." -ForegroundColor Cyan
        if (Test-Path $zip) { Remove-Item -Force $zip }
        Compress-Archive -Path (Join-Path $appDir "*") -DestinationPath $zip -Force
        Write-Host "     已生成: $zip" -ForegroundColor Green
    }
}
finally {
    Pop-Location
}
