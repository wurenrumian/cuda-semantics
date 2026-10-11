# EXP-0000-baseline

> 一域一实验（域 00）。定义与结论见 `docs/research/00-baseline.md`。
> 本实验**只读地收集本机环境事实**（GPU / 驱动 / CUDA / torch / 工具链路径），不编译内核、不改系统。

## 运行

```powershell
pwsh -File scripts/run.ps1
```

产物写入 `tmp/outputs/D00/<date>/`（本机，未入库）：`gpu.csv`、`nvcc.txt`、`torch.txt`、`tools.tsv`。

关键结论（GPU 型号 / compute_cap / nvcc 版本 / torch 是否可用）会同时打印到 stdout，站点「实验」页可直接回显。
