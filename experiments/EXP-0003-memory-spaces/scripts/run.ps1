# EXP-0003 runner: build + run memory-space demos + dump PTX/SASS.
# Usage: pwsh -File scripts/run.ps1
$exp    = Split-Path $PSScriptRoot -Parent
$tmp    = (Resolve-Path (Join-Path $exp '..\..\tmp')).Path
$out    = Join-Path $tmp 'outputs\D03\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$cs     = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'
$src    = Join-Path $PSScriptRoot 'mem_spaces.cu'
$exe    = Join-Path $out 'mem_spaces.exe'

Write-Host "build -> $exe"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -Xptxas -v -o `"$exe`" `"$src`"" *> (Join-Path $out 'build.log')

Write-Host "ptx"
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -arch=sm_89 -ptx -o `"$out\mem_spaces.ptx`" `"$src`"" *> (Join-Path $out 'ptx.log')

Write-Host "sass"
& $cuobj -sass $exe *> (Join-Path $out 'sass.txt')

# Per-function SASS opcode histogram: which memory-space opcodes each kernel uses.
$sassLines = Get-Content (Join-Path $out 'sass.txt')
$fn = '<none>'
$counts = [ordered]@{}
$ops = 'LDL', 'STL', 'LDS', 'STS', 'LDC', 'ULDC', 'LDG.E.CONSTANT', 'LDG', 'STG'
foreach ($ln in $sassLines) {
  if ($ln -match 'Function : (\S+)') {
    $fn = $Matches[1]
    if (-not $counts.Contains($fn)) { $counts[$fn] = @{} }
    continue
  }
  foreach ($op in $ops) {
    if ($ln -match ('\b' + [regex]::Escape($op) + '\b')) {
      if (-not $counts[$fn].ContainsKey($op)) { $counts[$fn][$op] = 0 }
      $counts[$fn][$op]++
    }
  }
}
$hist = foreach ($k in $counts.Keys) {
  $row = $counts[$k]
  [pscustomobject]@{
    function = $k
    LDL = [int]$row['LDL']; STL = [int]$row['STL']; LDS = [int]$row['LDS']
    STS = [int]$row['STS']; LDC = [int]$row['LDC']; ULDC = [int]$row['ULDC']
    RO = [int]$row['LDG.E.CONSTANT']; LDG = [int]$row['LDG']; STG = [int]$row['STG']
  }
}
$hist | Format-Table -AutoSize | Out-String | Set-Content (Join-Path $out 'sass-opcode-histogram.txt')

Write-Host "dump state-space usage from PTX"
$ptx = Join-Path $out 'mem_spaces.ptx'
# variable declarations: which state space each variable lives in
Select-String -Path $ptx -Pattern '^\s*\.(local|shared|const|global)\s' |
  ForEach-Object { $_.Line.Trim() } | Select-Object -Unique |
  ForEach-Object { "DECL  $_" } | Set-Content (Join-Path $out 'ptx-state-spaces.txt')
# memory instructions: which state space each load/store targets
Select-String -Path $ptx -Pattern '\b(ld|st|atom|red)\.(local|shared|const|global)\S*' -AllMatches |
  ForEach-Object { $_.Matches } | ForEach-Object { $_.Value } |
  Group-Object | Sort-Object Name |
  ForEach-Object { "MEM   {0,-28} x{1}" -f $_.Name, $_.Count } |
  Add-Content (Join-Path $out 'ptx-state-spaces.txt')

foreach ($m in 'attrs', 'local', 'chase', 'const', 'shared', 'coalesce', 'readonly') {
  Write-Host "run $m"
  & $exe $m *> (Join-Path $out "$m.txt")
}

Write-Host "compute-sanitizer memcheck: local OOB"
& $cs --tool memcheck $exe oob *> (Join-Path $out 'sanitizer-memcheck-oob.txt')
if ($LASTEXITCODE -ne 0) { Add-Content (Join-Path $out 'sanitizer-memcheck-oob.txt') "exitcode=$LASTEXITCODE" }

Write-Host "outputs -> $out"
