# EXP-0001 runner — SIMT 执行模型（mask / shuffle / reduce / ITS timing / sanitizer）
# 源: tmp/outputs/D01/src/d01_simt.cu（域 01 研究稿实际使用的 demo）。
# 产物统一写入 <repo>/tmp/outputs/EXP-0001/（本机，未入库）。
# Usage: pwsh -File scripts/run.ps1
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$repo = Split-Path (Split-Path (Split-Path $here -Parent) -Parent) -Parent
$out  = Join-Path $repo 'tmp\outputs\EXP-0001\2026-10-08'
$src  = Join-Path $here 'd01_simt.cu'
$bin  = Join-Path $out 'bin'
$log  = Join-Path $out 'logs'
New-Item -ItemType Directory -Force -Path $bin, $log | Out-Null

$nvcc  = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cs    = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'
$cuobj = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$exe   = Join-Path $bin 'd01_simt.exe'
$cubin = Join-Path $bin 'd01_simt.cubin'

Write-Host "build -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$src`"" *> (Join-Path $log 'build.log')
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -cubin -o `"$cubin`" `"$src`"" *> (Join-Path $log 'cubin.log')
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -ptx -o `"$(Join-Path $bin 'd01_simt.ptx')`" `"$src`"" *> (Join-Path $log 'ptx.log')

foreach ($m in 'mask','race-unsync','race-sync','shfl-bad','shfl-good','reduce-shfl','reduce-nosync','reduce-sync','timing','intrinsics','sass') {
  Write-Host "run $m"
  & $exe $m *> (Join-Path $log "$m.txt")
}

& $cuobj -sass -fun _Z6k_sassPii $cubin *> (Join-Path $log 'sass-k_sass.txt')
& $cuobj -sass -fun _Z8k_timingPyl $cubin *> (Join-Path $log 'sass-k_timing.txt')

Write-Host "sanitizer (racecheck/synccheck)"
& $cs --tool racecheck $exe race-unsync *> (Join-Path $log 'racecheck-unsync.txt')
& $cs --tool synccheck $exe shfl-bad    *> (Join-Path $log 'synccheck-shfl-bad.txt')

Write-Host "outputs -> $out"
