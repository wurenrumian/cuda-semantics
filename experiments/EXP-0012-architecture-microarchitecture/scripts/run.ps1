# EXP-0012 runner — architecture / microarchitecture probe (sm_89, CUDA 12.8).
# Usage: pwsh -File scripts/run.ps1
$exp    = Split-Path $PSScriptRoot -Parent
$tmp    = (Resolve-Path (Join-Path $exp '..\..\tmp')).Path
$out    = Join-Path $tmp 'outputs\D12\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$s      = $PSScriptRoot

function Invoke-Nvcc([string]$src, [string]$dst, [string]$arch, [string]$log, [string]$extra) {
  cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 $extra -arch=$arch -o `"$dst`" `"$src`"" *> $log
  return $LASTEXITCODE
}

Write-Host '1) arch_probe (props / occupancy / %smid / smem opt-in)'
Invoke-Nvcc (Join-Path $s 'arch_probe.cu') (Join-Path $out 'arch_probe.exe') 'sm_89' (Join-Path $out 'build-arch_probe.log') '' | Out-Null
& (Join-Path $out 'arch_probe.exe') *> (Join-Path $out 'arch_probe.out')

Write-Host '2) static shared >48KB (expect ptxas error)'
Invoke-Nvcc (Join-Path $s 'static_over.cu') (Join-Path $out 'static_over.exe') 'sm_89' (Join-Path $out 'build-static_over.log') '' | Out-Null

Write-Host '3) cp.async target gating (cubin: sm_75 fails, sm_89 ok)'
Invoke-Nvcc (Join-Path $s 'cpasync_gate.cu') (Join-Path $out 'cpasync_sm75.cubin') 'sm_75' (Join-Path $out 'build-cpasync_sm75.log') '-cubin' | Out-Null
Invoke-Nvcc (Join-Path $s 'cpasync_gate.cu') (Join-Path $out 'cpasync_sm89.cubin') 'sm_89' (Join-Path $out 'build-cpasync_sm89.log') '-cubin' | Out-Null
& $cuobj -sass (Join-Path $out 'cpasync_sm89.cubin') *> (Join-Path $out 'sass_cpasync_sm89.txt')

Write-Host '4) cross-arch SASS of a small kernel'
foreach ($a in 'sm_75', 'sm_80', 'sm_89') {
  cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -cubin -arch=$a -o `"$out\sass_probe_$a.cubin`" `"$s\sass_probe.cu`"" *> (Join-Path $out "build-sass_$a.log")
  & $cuobj -sass (Join-Path $out "sass_probe_$a.cubin") *> (Join-Path $out "sass_$a.txt")
}

Write-Host '5) arch suffix acceptance (sm_89a / sm_89f / sm_90a)'
foreach ($a in 'sm_89a', 'sm_89f', 'sm_90a') {
  Invoke-Nvcc (Join-Path $s 'sass_probe.cu') (Join-Path $out "suffix_$a.cubin") $a (Join-Path $out "build-suffix_$a.log") '-cubin' | Out-Null
}

Write-Host '6) coalescing / sector-granularity timing'
Invoke-Nvcc (Join-Path $s 'coalesce_probe.cu') (Join-Path $out 'coalesce_probe.exe') 'sm_89' (Join-Path $out 'build-coalesce.log') '' | Out-Null
foreach ($m in 'coalesced', 'strided') {
  & (Join-Path $out 'coalesce_probe.exe') $m 50 *> (Join-Path $out "coalesce-$m.txt")
  & (Join-Path $out 'coalesce_probe.exe') $m 50 *>> (Join-Path $out "coalesce-$m.txt")
}

Write-Host '7) ncu sector counters (expected: ERR_NVGPUCTRPERM on this host)'
$metrics = 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'
foreach ($m in 'coalesced', 'strided') {
  ncu --metrics $metrics --launch-count 1 (Join-Path $out 'coalesce_probe.exe') $m *> (Join-Path $out "ncu-$m.txt")
}

Write-Host "outputs -> $out"
