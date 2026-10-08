# EXP-0002 runner (domain 06 — synchronization): build + SASS dump + behavior.
# Usage: pwsh -File scripts/run.ps1
$exp    = Split-Path $PSScriptRoot -Parent
$tmp    = (Resolve-Path (Join-Path $exp '..\..\tmp')).Path
$out    = Join-Path $tmp 'outputs\D06\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cs     = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'
$objd   = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$src    = Join-Path $PSScriptRoot 'sync_demo.cu'
$bad    = Join-Path $PSScriptRoot 'badsync.cu'
$exe    = Join-Path $out 'sync_demo.exe'
$cubin  = Join-Path $out 'sync_demo.cubin'

& $nvcc --version *> (Join-Path $out 'nvcc-version.txt')

Write-Host "build exe   -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$src`"" *> (Join-Path $out 'build-exe.log')

Write-Host "build cubin -> $cubin"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -cubin -o `"$cubin`" `"$src`"" *> (Join-Path $out 'build-cubin.log')
& $objd -sass $cubin *> (Join-Path $out 'sass-full.txt')

Write-Host "ptxas probe: bar.sync 1,16 (expect error)"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -arch=sm_89 -cubin -o `"$out\badsync.cubin`" `"$bad`"" *> (Join-Path $out 'ptxas-badsync.txt')
"exit=$LASTEXITCODE" | Add-Content (Join-Path $out 'ptxas-badsync.txt')

Write-Host "plain runs"
& $exe smem-order-unsync *> (Join-Path $out 'run-smem-unsync.txt')
& $exe smem-order-sync   *> (Join-Path $out 'run-smem-sync.txt')
& $exe named-barrier     *> (Join-Path $out 'run-named-barrier.txt')
& $exe divergent-syncthreads     *> (Join-Path $out 'run-divergent-syncthreads-exit.txt')
& $exe divergent-syncwarp        *> (Join-Path $out 'run-divergent-syncwarp-exit.txt')
& $exe divergent-syncthreads-alive *> (Join-Path $out 'run-divergent-syncthreads-alive.txt')
& $exe syncwarp-partial-alive      *> (Join-Path $out 'run-syncwarp-partial-alive.txt')

Write-Host "compute-sanitizer template checks (may still fail to inject)"
& $cs --tool racecheck --error-exitcode 9 $exe smem-order-unsync *> (Join-Path $out 'racecheck-smem-unsync.txt')
& $cs --tool racecheck --error-exitcode 9 $exe smem-order-sync   *> (Join-Path $out 'racecheck-smem-sync.txt')
& $cs --tool synccheck --launch-timeout 30 --error-exitcode 9 $exe divergent-syncthreads *> (Join-Path $out 'synccheck-divergent-syncthreads.txt')
& $cs --tool synccheck --launch-timeout 30 --error-exitcode 9 $exe divergent-syncthreads-alive *> (Join-Path $out 'synccheck-divergent-syncthreads-alive.txt')

Write-Host "outputs -> $out"
