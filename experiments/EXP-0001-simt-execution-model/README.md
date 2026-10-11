# EXP-0001 — SIMT 执行模型

> 一域一实验。定义与结论见 `docs/research/01-simt-execution-model.md`。
> 源：域 01 研究稿实际使用的 demo：`scripts/d01_simt.cu`。

## 运行

```powershell
pwsh -File scripts/run.ps1
```

模式：`mask` / `race-unsync` / `race-sync` / `shfl-bad` / `shfl-good` /
`reduce-shfl` / `reduce-nosync` / `reduce-sync` / `timing` / `intrinsics` / `sass`。

补充实验（2026-10-11）：

- `pwsh -File scripts/run_activemask_good.ps1` —— `__activemask()` 的正当用法
  （`activemask_good.cu`）：发散分支内 Discovery Pattern（快照 → `__syncwarp(快照)` → 稳定集合 →
  集合内 shuffle）与全员收敛对照，产物 `tmp/outputs/EXP-0001/2026-10-11/activemask-good/`，
  对应研究稿「观察 4」。

产物写入 `tmp/outputs/EXP-0001/`（本机，未入库）。`activemask_good` 产物在
`tmp/outputs/EXP-0001/2026-10-11/activemask-good/`。
