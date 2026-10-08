# EXP-0009 runner: build + run async/pipeline demos, dump SASS, probe TMA,
# capture everything under tmp/outputs/D09/2026-10-08/.
# Usage: pwsh -File scripts/run.ps1
$ErrorActionPreference = 'Continue'
$exp   = Split-Path $PSScriptRoot -Parent
$tmp   = (Resolve-Path (Join-Path $exp '..\..\tmp')).Path
$out   = Join-Path $tmp 'outputs\D09\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $out 'sass') | Out-Null

$nvcc    = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars  = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cuobj   = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$cs      = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'
$src     = Join-Path $PSScriptRoot 'async_pipelines.cu'
$tma     = Join-Path $PSScriptRoot 'tma_probe.cu'
$exe     = Join-Path $out 'async_pipelines.exe'
$exeDS   = Join-Path $out 'ds_legacy.exe'
$exeDSp  = Join-Path $out 'ds_perthread.exe'

function Build($outfile, $extra, $logname, $source = $src, $arch = 'sm_89') {
  $flags = "-O3 -lineinfo -arch=$arch $extra"
  cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" $flags -o `"$outfile`" `"$source`"" *> (Join-Path $out $logname)
}

Write-Host "build main"
Build $exe '' 'build-main.log'
Write-Host "build ds legacy"
Build $exeDS '-default-stream legacy' 'build-ds-legacy.log'
Write-Host "build ds per-thread"
Build $exeDSp '--default-stream per-thread' 'build-ds-perthread.log'

Write-Host "run modes"
& $exe props        *> (Join-Path $out 'props.txt')
& $exe streams      *> (Join-Path $out 'streams.txt')
& $exe memcpy       *> (Join-Path $out 'memcpy.txt')
& $exe cpasync      *> (Join-Path $out 'cpasync.txt')
& $exe cpasync-racy *> (Join-Path $out 'cpasync-racy.txt')
& $exe graph        *> (Join-Path $out 'graph.txt')

Write-Host "run ds scenarios"
& $exeDS  ds *> (Join-Path $out 'default-stream-legacy.txt')
& $exeDSp ds *> (Join-Path $out 'default-stream-perthread.txt')

Write-Host "sass dump"
& $cuobj -sass $exe *> (Join-Path $out 'sass\async_pipelines.sass')

# --- extract the three copy kernels side by side ---------------------------
$sass = Get-Content (Join-Path $out 'sass\async_pipelines.sass')
$blocks = @{}
$cur = $null
foreach ($line in $sass) {
  if ($line -match 'Function : (\S+)') { $cur = $Matches[1]; $blocks[$cur] = @() }
  elseif ($cur) { $blocks[$cur] += ($line -replace '\s+/\* 0x[0-9a-f]+\s+\*/', '') }
}
$report = New-Object System.Collections.Generic.List[string]
foreach ($fn in @('_Z12cpasync_copyPK6float4PS_i','_Z12cpasync_racyPK6float4PS_i','_Z11manual_copyPK6float4PS_i')) {
  if ($blocks.ContainsKey($fn)) {
    $report.Add("===== $fn =====")
    $blocks[$fn] | Where-Object { $_ -match 'LDGSTS|LDG|STS|LDS|BAR|DEPBAR|CPASYNC' } | ForEach-Object { $report.Add($_.Trim()) }
    $report.Add('')
  }
}
$report | Set-Content (Join-Path $out 'sass\cpasync_sass_compare.txt')

Write-Host "tma probe (sm_89 expected FAIL, sm_90 expected OK)"
Build (Join-Path $out 'tma_sm89.cubin') '-cubin' 'tma-sm89.log' $tma 'sm_89'
Build (Join-Path $out 'tma_sm90.cubin') '-cubin' 'tma-sm90.log' $tma 'sm_90'

Write-Host "compute-sanitizer racecheck on cpasync (NOTE: known-broken on this box)"
& $cs --tool racecheck --error-exitcode 9 $exe cpasync *> (Join-Path $out 'racecheck-cpasync.txt')

Write-Host "outputs -> $out"
