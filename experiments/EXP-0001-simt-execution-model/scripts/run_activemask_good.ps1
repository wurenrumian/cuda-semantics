# EXP-0001 补充 runner：__activemask() 正当用法（未决 ⑤）
# Usage: pwsh -File run_activemask_good.ps1
$src  = $PSScriptRoot
$root = (Resolve-Path (Join-Path $src '..\..\..\tmp')).Path
$out  = Join-Path $root 'outputs\EXP-0001\2026-10-11\activemask-good'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cu     = Join-Path $src 'activemask_good.cu'
$exe    = Join-Path $out 'activemask_good.exe'

Write-Host "build -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$cu`"" *> (Join-Path $out 'build.log')
if (-not (Test-Path $exe)) { Write-Host "BUILD FAILED"; Get-Content (Join-Path $out 'build.log'); exit 1 }

Write-Host "run discovery"
& $exe discovery  *> (Join-Path $out 'discovery.txt')
Get-Content (Join-Path $out 'discovery.txt')

Write-Host "run baseline"
& $exe baseline   *> (Join-Path $out 'baseline.txt')
Get-Content (Join-Path $out 'baseline.txt')

Write-Host "outputs -> $out"
