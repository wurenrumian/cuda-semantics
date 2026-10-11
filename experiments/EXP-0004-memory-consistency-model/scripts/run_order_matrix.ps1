# EXP-0004 补充 runner：order × scope 矩阵（PTX + SASS + smoke）
# Usage: pwsh -File run_order_matrix.ps1
$src  = $PSScriptRoot
$root = (Resolve-Path (Join-Path $src '..\..\..\tmp')).Path
$out  = Join-Path $root 'outputs\D04\2026-10-11\order-matrix'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cu     = Join-Path $src 'order_matrix.cu'
$exe    = Join-Path $out 'order_matrix.exe'
$cubin  = Join-Path $out 'order_matrix.cubin'

Write-Host "build -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$cu`"" *> (Join-Path $out 'build.log')
if (-not (Test-Path $exe)) { Write-Host "BUILD FAILED"; Get-Content (Join-Path $out 'build.log'); exit 1 }

Write-Host "run smoke"
& $exe *> (Join-Path $out 'smoke.txt')
Get-Content (Join-Path $out 'smoke.txt')

Write-Host "cubin + sass"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -cubin -o `"$cubin`" `"$cu`"" 2> (Join-Path $out 'cubin.log')
if (Test-Path $cubin) {
  & $cuobj -sass $cubin *> (Join-Path $out 'sass.txt')
}

Write-Host "ptx"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -ptx -o `"$out\order_matrix.ptx`" `"$cu`"" 2> (Join-Path $out 'ptx.log')

Write-Host "outputs -> $out"
