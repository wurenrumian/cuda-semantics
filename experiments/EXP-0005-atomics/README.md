# EXP-0005 — 原子语义

> 一域一实验。定义与结论见 `docs/research/05-atomics.md`。

## 运行

```powershell
pwsh -File scripts/run.ps1
```

模式：`props` / `atomicity` / `race-naive` / `race-atomic` /
`publish <0|1> <rounds>` / `floatmax`。

> `publish` 的消费者用**有界自旋**（超时即报告，不挂住）；跨 grid/kernel 的生产者-消费者
> 本身**不保证同时驻留**，超时只是教学保护，不代表算法安全。详见研究稿反例 1。

产物写入 `tmp/outputs/D05/`（本机，未入库）。
