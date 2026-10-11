# EXP-0004 — 内存一致性模型

> 一域一实验。定义与结论见 `docs/research/04-memory-consistency-model.md`。

## 运行

```powershell
pwsh -File scripts/run.ps1
```

模式：`volatile-nonatomic` / `atomic` / `producers` / `msg-naive` / `msg-acqrel`。

> `msg-*` 是跨 block 的生产者-消费者：消费者自旋有上界（`spin_max`），超时会计入
> `timeouts` 而不是挂住。CUDA **不保证**两个 block 同时驻留，所以跨 block 自旋本身
> 需要驻留/顺序保证。详见研究稿反例 ④。

产物写入 `tmp/outputs/D04/2026-10-08/`（本机，未入库）。
