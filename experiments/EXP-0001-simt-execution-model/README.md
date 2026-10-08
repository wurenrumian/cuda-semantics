# EXP-0001 — SIMT 执行模型

> 一域一实验。定义与结论见 `docs/research/01-simt-execution-model.md`。
> 源：域 01 研究稿实际使用的 demo（原 `tmp/outputs/D01/src/d01_simt.cu`）。

## 运行

```powershell
pwsh -File scripts/run.ps1
```

模式：`mask` / `race-unsync` / `race-sync` / `shfl-bad` / `shfl-good` /
`reduce-shfl` / `reduce-nosync` / `reduce-sync` / `timing` / `intrinsics` / `sass`。

产物写入 `tmp/outputs/EXP-0001/`（本机，未入库）。
