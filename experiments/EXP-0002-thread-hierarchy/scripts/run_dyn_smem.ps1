# EXP-0002 补充 runner：动态 shared memory × launch_bounds
# Usage: pwsh -File run_dyn_smem.ps1
$src  = $PSScriptRoot
$root = (Resolve-Path (Join-Path $src '..\..\..\tmp')).Path
$out  = Join-Path $root 'outputs\D02\2026-10-11\dyn-smem'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cu     = Join-Path $src 'dyn_smem.cu'
$exe    = Join-Path $out 'dyn_smem.exe'

Write-Host "build -> $exe（-Xptxas -v 抓寄存器与 spill）"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -Xptxas -v -o `"$exe`" `"$cu`"" *> (Join-Path $out 'build.log')
if (-not (Test-Path $exe)) { Write-Host "BUILD FAILED"; Get-Content (Join-Path $out 'build.log'); exit 1 }

Write-Host "run"
& $exe *> (Join-Path $out 'res.txt')
Get-Content (Join-Path $out 'res.txt')

Write-Host "outputs -> $out"
