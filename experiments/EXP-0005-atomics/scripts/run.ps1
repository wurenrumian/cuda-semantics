# D05 atomics lab runner: build, run demos, inspect PTX/SASS, run sanitizer.
# Usage: pwsh -File run.ps1
$script = $PSScriptRoot
$repo   = Split-Path (Split-Path (Split-Path $script -Parent) -Parent) -Parent
$out    = Join-Path $repo 'tmp\outputs\D05'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cs     = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'

$lab  = Join-Path $script 'atomics_lab.cu'
$sass = Join-Path $script 'atomics_sass.cu'
$exe  = Join-Path $out 'atomics_lab.exe'

$pre = "call `"$vcvars`" >nul 2>&1 && "

Write-Host "build lab -> $exe"
cmd /c "$pre`"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$lab`"" *> (Join-Path $out 'build.log')

foreach ($m in @('props','atomicity','race-naive','race-atomic','publish 0 200','publish 1 200','publish 0 2000','floatmax')) {
  $tag = ($m -replace ' ','_')
  Write-Host "run $m"
  cmd /c "`"$exe`" $m" *> (Join-Path $out "run_$tag.txt")
}

Write-Host "PTX / SASS"
cmd /c "$pre`"$nvcc`" -arch=sm_89 -ptx   -o `"$out\atomics_sass.ptx`"   `"$sass`"" *> (Join-Path $out 'ptx.log')
cmd /c "$pre`"$nvcc`" -arch=sm_89 -cubin -o `"$out\atomics_sass.cubin`" `"$sass`"" *> (Join-Path $out 'cubin.log')
& $cuobj -sass (Join-Path $out 'atomics_sass.cubin') *> (Join-Path $out 'atomics_sass.sass')

Write-Host "boundary: 128-bit atomic on sm_89 (expected to fail)"
cmd /c "$pre`"$nvcc`" -arch=sm_89 -cubin -o `"$out\atomics_b128_sm89.cubin`" `"$script\atomics_b128_sm90.cu`"" *> (Join-Path $out 'b128_sm89_build.log')
cmd /c "$pre`"$nvcc`" -arch=sm_90 -cubin -o `"$out\atomics_b128_sm90.cubin`" `"$script\atomics_b128_sm90.cu`"" *> (Join-Path $out 'b128_sm90_build.log')

Write-Host "racecheck: naive"
& $cs --tool racecheck --error-exitcode 9 $exe race-naive *> (Join-Path $out 'racecheck_naive.txt')
Write-Host "racecheck: atomic"
& $cs --tool racecheck --error-exitcode 9 $exe race-atomic *> (Join-Path $out 'racecheck_atomic.txt')

Write-Host "outputs -> $out"
