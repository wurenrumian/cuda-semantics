# EXP-0000 runner — environment baseline probe (sm_89, CUDA 12.8).
# Read-only: collects GPU / driver / CUDA / torch / toolchain facts. No kernel build.
# Usage: pwsh -File scripts/run.ps1
$exp = Split-Path $PSScriptRoot -Parent
$tmp = [IO.Path]::GetFullPath((Join-Path $exp '..\..\tmp'))
$out = Join-Path $tmp ('outputs\D00\' + (Get-Date -Format 'yyyy-MM-dd'))
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nvcc = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$py   = 'E:\venvs\pytorch312\Scripts\python.exe'

Write-Host '1) GPU / driver / compute capability'
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv |
  Tee-Object -FilePath (Join-Path $out 'gpu.csv')

Write-Host '2) CUDA toolkit'
& $nvcc --version | Tee-Object -FilePath (Join-Path $out 'nvcc.txt')

Write-Host '3) PyTorch'
& $py -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),torch.cuda.get_device_name(0))" *>&1 |
  Tee-Object -FilePath (Join-Path $out 'torch.txt')

Write-Host '4) toolchain paths'
$tools = [ordered]@{
  'nvcc'              = $nvcc
  'ptxas'             = 'E:\NVIDIA\CUDA\v12.8\bin\ptxas.exe'
  'cuobjdump'         = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
  'nvdisasm'          = 'E:\NVIDIA\CUDA\v12.8\bin\nvdisasm.exe'
  'compute-sanitizer' = 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'
  'vcvars64'          = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
}
$rows = foreach ($k in $tools.Keys) {
  $ok = Test-Path $tools[$k]
  Write-Host ("  {0,-18} {1,-6} {2}" -f $k, $ok, $tools[$k])
  "{0}`t{1}`t{2}" -f $k, $ok, $tools[$k]
}
$rows | Set-Content -Encoding utf8 (Join-Path $out 'tools.tsv')

Write-Host "outputs -> $out"
