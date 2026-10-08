# EXP-0008 runner: 四组浮点编译选项 + PTX/SASS 对照 + 求和可复现性。
# Usage: pwsh -File scripts/run.ps1
$exp    = Split-Path $PSScriptRoot -Parent
$tmp    = (Resolve-Path (Join-Path $exp '..\..\tmp')).Path
$out    = Join-Path $tmp 'outputs\D08\2026-10-08'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$src    = Join-Path $PSScriptRoot 'fp_demo.cu'
$arch   = 'sm_89'

function Build($name, $flags) {
  $exe = Join-Path $out "$name.exe"
  Write-Host "build $name : $flags"
  cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" $flags -o `"$exe`" `"$src`"" `
    2>&1 | Out-File -Encoding utf8 (Join-Path $out "build-$name.log")
  return $exe
}

$default  = Build 'fp_default'  "-O3 -lineinfo -arch=$arch"
$nofmad   = Build 'fp_nofmad'   "-O3 -lineinfo -arch=$arch -fmad=false"
$fastmath = Build 'fp_fastmath' "-O3 -lineinfo -arch=$arch -use_fast_math"
$ftz      = Build 'fp_ftz'      "-O3 -lineinfo -arch=$arch --ftz=true"
$debug    = Build 'fp_debug'    "-G -lineinfo -arch=$arch"

foreach ($pair in @(
    @($default,  'default'),
    @($nofmad,   'nofmad'),
    @($fastmath, 'fastmath'),
    @($ftz,      'ftz'),
    @($debug,    'debug'))) {
  Write-Host "run fp [$($pair[1])]"
  & $pair[0] fp 2>&1 | Out-File -Encoding utf8 (Join-Path $out "fp-$($pair[1]).txt")
}

# PTX / SASS 对照（default vs nofmad vs fastmath）
foreach ($pair in @(
    @($default,  'default'),
    @($nofmad,   'nofmad'),
    @($fastmath, 'fastmath'))) {
  $exe = $pair[0]; $tag = $pair[1]
  & $cuobj -sass $exe 2>&1 | Out-File -Encoding utf8 (Join-Path $out "sass-$tag.txt")
}
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=$arch -ptx -o `"$(Join-Path $out 'fp_default.ptx')`" `"$src`""
cmd /c "call `"$vcvars`" >nul 2>&1 && `"$nvcc`" -O3 -lineinfo -arch=$arch -use_fast_math -ptx -o `"$(Join-Path $out 'fp_fastmath.ptx')`" `"$src`""

# 求和可复现性：默认编译跑 5 次
Write-Host "run sum x5"
& $default sum 5 2>&1 | Out-File -Encoding utf8 (Join-Path $out 'sum-atomic.txt')

# 原子浮点 denormal：global vs shared
Write-Host "run atomden [default]"
& $default atomden 2>&1 | Out-File -Encoding utf8 (Join-Path $out 'atomden-default.txt')

# 逐值探针：fast math 下 1/q 与 sqrt(q) 的 bit 差异
Write-Host "run probe [default/fastmath]"
& $default  probe 2>&1 | Out-File -Encoding utf8 (Join-Path $out 'probe-default.txt')
& $fastmath probe 2>&1 | Out-File -Encoding utf8 (Join-Path $out 'probe-fastmath.txt')

Write-Host "outputs -> $out"
