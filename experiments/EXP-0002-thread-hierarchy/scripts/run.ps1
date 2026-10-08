# EXP-0002 runner: build + scheduled-order / occupancy demos + SASS + synccheck.
# Usage: pwsh -File scripts/run.ps1
$exp   = Split-Path $PSScriptRoot -Parent
$tmp   = (Resolve-Path (Join-Path $exp '..\..\tmp')).Path
$out   = Join-Path $tmp 'outputs\D02\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cs     = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$src    = Join-Path $PSScriptRoot 'thread_hierarchy.cu'
$exe    = Join-Path $out 'thread_hierarchy.exe'

Write-Host "build -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -Xptxas -v -o `"$exe`" `"$src`"" *> (Join-Path $out 'build.log')

Write-Host "run sched"
& $exe sched *> (Join-Path $out 'sched.txt')

Write-Host "run occupancy"
& $exe occupancy *> (Join-Path $out 'occupancy.txt')

Write-Host "cuobjdump sass"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$cuobj`" -sass `"$exe`"" *> (Join-Path $out 'sass.txt')

Write-Host "compute-sanitizer synccheck (barrier)"
$so  = Join-Path $out 'synccheck.txt'
$se  = Join-Path $out 'synccheck.err.txt'
$p = Start-Process -FilePath $cs -ArgumentList @('--tool','synccheck',"$exe",'barrier') `
     -NoNewWindow -PassThru -RedirectStandardOutput $so -RedirectStandardError $se
if (-not $p.WaitForExit(90000)) {
  $p.Kill()
  Add-Content $so "`n[TIMEOUT: synccheck did not return within 90s; killed]"
}

Write-Host "outputs -> $out"
