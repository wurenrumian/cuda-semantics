# EXP-0006 runner (domain 06 — synchronization): build + SASS dump + behavior.
# Usage: pwsh -File scripts/run.ps1
#
# Safety: the "*-alive" demos wedge on purpose (deliberate non-terminating spin
# at a divergent barrier).  sync_demo.exe self-terminates via its in-process
# watchdog (~0.5 s deadline; measured wall time is printed), and every GPU run
# below is additionally wrapped in a hard process timeout so this script can
# never hang the shell.
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

# Run a demo exe with a hard wall-clock timeout (fallback watchdog); the target
# binary itself also has an in-process watchdog, this is belt-and-suspenders.
function Invoke-GpuExe {
  param([string]$ExePath, [string]$Mode, [string]$LogPath, [int]$TimeoutSec = 30)
  $err = "$LogPath.err"
  $p = Start-Process -FilePath $ExePath -ArgumentList $Mode -NoNewWindow -PassThru `
       -RedirectStandardOutput $LogPath -RedirectStandardError $err
  if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    $p.Kill()
    Add-Content -Path $LogPath -Value "HARD TIMEOUT: killed after ${TimeoutSec}s (watchdog fallback)"
  }
  $p.WaitForExit()          # drain redirected streams
  if (Test-Path $err) { Get-Content $err | Add-Content -Path $LogPath; Remove-Item $err -Force }
}

& $nvcc --version *> (Join-Path $out 'nvcc-version.txt')

Write-Host "build exe   -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$exe`" `"$src`"" *> (Join-Path $out 'build-exe.log')

Write-Host "build cubin -> $cubin"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -cubin -o `"$cubin`" `"$src`"" *> (Join-Path $out 'build-cubin.log')
& $objd -sass $cubin *> (Join-Path $out 'sass-full.txt')
# Key barrier lines only (what domain 06 reads in the doc).
Select-String -Path (Join-Path $out 'sass-full.txt') `
  -Pattern 'Function :|BAR\.SYNC|WARPSYNC|BRA\.CONV|EXIT' |
  ForEach-Object { $_.Line } | Set-Content (Join-Path $out 'sass-key.txt')

Write-Host "ptxas probe: bar.sync 1,16 (expect error)"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -arch=sm_89 -cubin -o `"$out\badsync.cubin`" `"$bad`"" *> (Join-Path $out 'ptxas-badsync.txt')
"exit=$LASTEXITCODE" | Add-Content (Join-Path $out 'ptxas-badsync.txt')

Write-Host "plain runs (bounded by a 30 s hard timeout each)"
Invoke-GpuExe $exe 'smem-order-unsync'           (Join-Path $out 'run-smem-unsync.txt')
Invoke-GpuExe $exe 'smem-order-sync'             (Join-Path $out 'run-smem-sync.txt')
Invoke-GpuExe $exe 'named-barrier'               (Join-Path $out 'run-named-barrier.txt')
Invoke-GpuExe $exe 'divergent-syncthreads'       (Join-Path $out 'run-divergent-syncthreads-exit.txt')
Invoke-GpuExe $exe 'divergent-syncwarp'          (Join-Path $out 'run-divergent-syncwarp-exit.txt')
Invoke-GpuExe $exe 'divergent-syncthreads-alive' (Join-Path $out 'run-divergent-syncthreads-alive.txt')
Invoke-GpuExe $exe 'syncwarp-partial-alive'      (Join-Path $out 'run-syncwarp-partial-alive.txt')

Write-Host "compute-sanitizer template checks (may still fail to inject)"
& $cs --tool racecheck --launch-timeout 30 --error-exitcode 9 $exe smem-order-unsync *> (Join-Path $out 'racecheck-smem-unsync.txt')
& $cs --tool racecheck --launch-timeout 30 --error-exitcode 9 $exe smem-order-sync   *> (Join-Path $out 'racecheck-smem-sync.txt')
& $cs --tool synccheck --launch-timeout 30 --error-exitcode 9 $exe divergent-syncthreads *> (Join-Path $out 'synccheck-divergent-syncthreads.txt')
& $cs --tool synccheck --launch-timeout 30 --error-exitcode 9 $exe divergent-syncthreads-alive *> (Join-Path $out 'synccheck-divergent-syncthreads-alive.txt')

Write-Host "outputs -> $out"
