# D07 指令级/PTX — 本机验证脚本 (sm_89 / CUDA 12.8)
# 用法: powershell -ExecutionPolicy Bypass -File run.ps1
$ErrorActionPreference = 'Stop'
$vs    = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cuda  = 'E:\NVIDIA\CUDA\v12.8\bin'
$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path (Split-Path (Split-Path $here -Parent) -Parent) -Parent
$out  = Join-Path $repo 'tmp\outputs\EXP-0007\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Dev([string]$cmd) {
    Write-Host ">> $cmd" -ForegroundColor DarkGray
    cmd /c "call `"$vs`" >nul 2>nul && $cmd"
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $cmd" }
}

$srcs = '01_cache_ops','02_strong_vs_weak','03_ldg_vs_plain','04_restrict_alias','05_inline_ptx','06_ptx_cache'

foreach ($s in $srcs) {
    $cu = Join-Path $here "$s.cu"
    # 1) 可执行：跑起来看运行结果
    Dev "`"$cuda\nvcc.exe`" -arch=sm_89 -O3 -lineinfo `"$cu`" -o `"$out\$s.exe`""
    # 2) PTX：看指令级语义
    Dev "`"$cuda\nvcc.exe`" -arch=sm_89 -O3 -ptx `"$cu`" -o `"$out\$s.ptx`""
    # 3) SASS：看硬件指令（弱/强访存、cache op 编码）
    Dev "`"$cuda\nvcc.exe`" -arch=sm_89 -O3 -cubin `"$cu`" -o `"$out\$s.cubin`""
    Dev "`"$cuda\cuobjdump.exe`" -sass `"$out\$s.cubin`" > `"$out\$s.sass`""
}

Write-Host "=== 运行结果 ===" -ForegroundColor Cyan
foreach ($s in $srcs) {
    Write-Host "--- $s ---" -ForegroundColor Yellow
    & (Join-Path $out "$s.exe")
}
