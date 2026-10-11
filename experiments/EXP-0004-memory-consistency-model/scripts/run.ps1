# D04 memory-consistency runner: build + run probes + PTX/SASS + sanitizer.
# Usage: pwsh -File run.ps1
$src  = $PSScriptRoot
$root = (Resolve-Path (Join-Path $src '..\..\..\tmp')).Path         # scripts -> EXP-0004 -> experiments -> repo, then \tmp
$out  = Join-Path $root 'outputs\D04\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cs     = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'
$cu     = Join-Path $src 'memmodel_demo.cu'
$exe    = Join-Path $out 'memmodel_demo.exe'

Write-Host "build -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$cu`"" *> (Join-Path $out 'build.log')
if (-not (Test-Path $exe)) { Write-Host "BUILD FAILED"; Get-Content (Join-Path $out 'build.log'); exit 1 }

Write-Host "run volatile-nonatomic"
& $exe volatile-nonatomic 4000 *> (Join-Path $out 'volatile-nonatomic.txt')
Write-Host "run atomic"
& $exe atomic 4000 *> (Join-Path $out 'atomic.txt')
Write-Host "run producers"
& $exe producers *> (Join-Path $out 'producers.txt')
Write-Host "run msg-naive"
& $exe msg-naive 2000 *> (Join-Path $out 'msg-naive.txt')
Write-Host "run msg-acqrel"
& $exe msg-acqrel 2000 *> (Join-Path $out 'msg-acqrel.txt')

Write-Host "ptx"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -ptx -o `"$out\memmodel_demo.ptx`" `"$cu`"" 2> (Join-Path $out 'ptx.log')

Write-Host "cubin + sass"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -cubin -o `"$out\memmodel_demo.cubin`" `"$cu`"" 2> (Join-Path $out 'cubin.log')
if (Test-Path (Join-Path $out 'memmodel_demo.cubin')) {
  & $cuobj -sass (Join-Path $out 'memmodel_demo.cubin') *> (Join-Path $out 'sass.txt')
}

Write-Host "racecheck: volatile-nonatomic"
& $cs --tool racecheck --error-exitcode 9 $exe volatile-nonatomic 2000 *> (Join-Path $out 'racecheck-volatile.txt')
Write-Host "racecheck: atomic"
& $cs --tool racecheck --error-exitcode 9 $exe atomic 2000 *> (Join-Path $out 'racecheck-atomic.txt')

Write-Host "outputs -> $out"
