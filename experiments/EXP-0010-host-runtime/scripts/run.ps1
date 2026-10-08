# EXP-0010 runner: build + run all host-runtime demos, capture output.
# Usage: pwsh -File scripts/run.ps1
$exp  = Split-Path $PSScriptRoot -Parent
$tmp  = (Resolve-Path (Join-Path $exp '..\..\tmp')).Path
$out  = Join-Path $tmp 'outputs\D10\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc  = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cs    = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'

$src = Join-Path $PSScriptRoot 'runtime_demo.cu'
$exe = Join-Path $out 'runtime_demo.exe'
$dpsrc = Join-Path $PSScriptRoot 'dp_parent.cu'
$dpexe = Join-Path $out 'dp_parent.exe'

Write-Host "build runtime_demo -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$src`"" *> (Join-Path $out 'build.log')

foreach ($m in 'config','peekget','async','streams','props') {
  Write-Host "run $m"
  & $exe $m *> (Join-Path $out "$m.txt")
}

Write-Host "racecheck/initcheck irrelevant here; run memcheck on async fault"
& $cs --tool memcheck $exe async *> (Join-Path $out 'memcheck-async.txt')

Write-Host "build dp_parent WITHOUT rdc (expected to fail) -> dp_build_nordc.log"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -o `"$out\dp_nordc.exe`" `"$dpsrc`"" *> (Join-Path $out 'dp_build_nordc.log')

Write-Host "build dp_parent WITH rdc -> dp_parent.exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -rdc=true -lcudadevrt -o `"$dpexe`" `"$dpsrc`"" *> (Join-Path $out 'dp_build_rdc.log')
if (Test-Path $dpexe) { Write-Host "run dp_parent"; & $dpexe *> (Join-Path $out 'dp_run.txt') }

# dynamic shared memory opt-in boundary
$smemsrc = Join-Path $PSScriptRoot 'smem_optin.cu'
Write-Host "build+run smem_optin"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -o `"$out\smem_optin.exe`" `"$smemsrc`"" *> (Join-Path $out 'smem_build.log')
if (Test-Path (Join-Path $out 'smem_optin.exe')) { & (Join-Path $out 'smem_optin.exe') *> (Join-Path $out 'smem_optin.txt') }

# __launch_bounds__ -> PTX entry directives
$lbsrc = Join-Path $PSScriptRoot 'launchbounds.cu'
Write-Host "emit PTX launchbounds.ptx"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -arch=sm_89 -ptx `"$lbsrc`" -o `"$out\launchbounds.ptx`"" *> (Join-Path $out 'launchbounds_build.log')

# device-side observation: SASS + resource usage of runtime_demo
$cuobjdump = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
Write-Host "cuobjdump -sass / -res-usage"
& $cuobjdump -sass $exe *> (Join-Path $out 'cuobjdump_sass.txt')
& $cuobjdump -res-usage $exe *> (Join-Path $out 'cuobjdump_resusage.txt')

Write-Host "outputs -> $out"
