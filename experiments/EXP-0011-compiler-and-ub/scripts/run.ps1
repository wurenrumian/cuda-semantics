# Domain 11 runner: build (O3 + G) + SASS dumps + runs + sanitizers.
# Usage: pwsh -File run.ps1
$here  = $PSScriptRoot
$repo  = Split-Path (Split-Path (Split-Path $here -Parent) -Parent) -Parent
$out   = Join-Path $repo 'tmp\outputs\EXP-0011\2026-10-08'
$src   = Join-Path $here 'd11_demos.cu'
$exe   = Join-Path $out 'd11_demos.exe'
$exeG  = Join-Path $out 'd11_demos_G.exe'

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$cub    = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cs     = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'

New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host "build O3 -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$src`"" *> (Join-Path $out 'build-O3.log')

Write-Host "build -G -O0 -> $exeG"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -G -O0 -arch=sm_89 -o `"$exeG`" `"$src`"" *> (Join-Path $out 'build-G.log')

Write-Host "sass (O3 / -G)"
& $cub -sass $exe  *> (Join-Path $out 'sass-O3.txt')
& $cub -sass $exeG *> (Join-Path $out 'sass-G.txt')

Write-Host "run demos (O3)"
& $exe counter-volatile *> (Join-Path $out 'run-counter-volatile.txt')
& $exe counter-atomic   *> (Join-Path $out 'run-counter-atomic.txt')
& $exe restrict         *> (Join-Path $out 'run-restrict.txt')
& $exe deadstore        *> (Join-Path $out 'run-deadstore.txt')
& $exe uninit           *> (Join-Path $out 'run-uninit.txt')

Write-Host "sanitizers"
& $cs --tool racecheck $exe counter-shared *> (Join-Path $out 'racecheck-counter-shared.txt')
& $cs --tool initcheck $exe uninit *> (Join-Path $out 'initcheck-uninit.txt')

Write-Host "outputs -> $out"
