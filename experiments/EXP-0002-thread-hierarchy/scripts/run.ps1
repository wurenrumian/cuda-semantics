# EXP-0002 runner: build + scheduled-order / occupancy / barrier demos + SASS + synccheck.
# Every GPU-program run (sched / occupancy / barrier / sanitizer) runs under a host-side
# watchdog: a mode that wedges the GPU is killed together with its whole process tree
# (compute-sanitizer's target is a child of the sanitizer) and marked [WATCHDOG], so this
# script (and the local runner driving it) always returns instead of hanging forever.
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

# Run $File with $Arg, capturing stdout/stderr; kill it after $TimeoutSec and record a
# [WATCHDOG] line in $Stdout. Returns $true when the child exited on its own.
function Invoke-Watched {
  param([string]$File, [string[]]$Arg, [string]$Stdout, [string]$Stderr, [int]$TimeoutSec)
  $p = Start-Process -FilePath $File -ArgumentList $Arg -NoNewWindow -PassThru `
       -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
  if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    & taskkill.exe /T /F /PID $p.Id *>$null
    $p.WaitForExit(5000) | Out-Null
    Add-Content $Stdout "`n[WATCHDOG: $File $($Arg -join ' ') did not return within ${TimeoutSec}s -> killed (mode wedged)]"
    return $false
  }
  if ($p.ExitCode -ne 0) { Add-Content $Stdout "`n[exit code $($p.ExitCode)]" }
  return $true
}

Write-Host "build -> $exe"
Remove-Item $exe -ErrorAction SilentlyContinue
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -Xptxas -v -o `"$exe`" `"$src`"" *> (Join-Path $out 'build.log')
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $exe)) { Write-Host "build failed, see build.log"; exit 1 }

Write-Host 'run sched'
Invoke-Watched -File $exe -Arg @('sched') -Stdout (Join-Path $out 'sched.txt') `
  -Stderr (Join-Path $out 'sched.err.txt') -TimeoutSec 120 | Out-Null

Write-Host 'run occupancy'
Invoke-Watched -File $exe -Arg @('occupancy') -Stdout (Join-Path $out 'occupancy.txt') `
  -Stderr (Join-Path $out 'occupancy.err.txt') -TimeoutSec 120 | Out-Null

Write-Host 'run barrier'
Invoke-Watched -File $exe -Arg @('barrier') -Stdout (Join-Path $out 'barrier.txt') `
  -Stderr (Join-Path $out 'barrier.err.txt') -TimeoutSec 60 | Out-Null

Write-Host 'cuobjdump sass'
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$cuobj`" -sass `"$exe`"" *> (Join-Path $out 'sass.txt')

Write-Host 'compute-sanitizer synccheck (expected to stay blocked on this machine)'
Invoke-Watched -File $cs -Arg @('--tool','synccheck',$exe,'barrier') `
  -Stdout (Join-Path $out 'sanitizer-blocked.txt') `
  -Stderr (Join-Path $out 'sanitizer-blocked.err.txt') -TimeoutSec 90 | Out-Null

Write-Host "outputs -> $out"
