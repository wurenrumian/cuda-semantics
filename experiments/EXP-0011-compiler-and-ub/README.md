# EXP-0011-compiler-and-ub

> 一域一实验。定义与结论见 `docs/research/11-compiler-and-ub.md`。

## 运行

```powershell
pwsh -File scripts/run.ps1
```

产物写入 `tmp/outputs/D11/`（本机，未入库）。

## 运行记录

| 日期 | 内容 | 产物 |
|---|---|---|
| 2026-10-08 | run.ps1（restrict/counter/deadstore/uninit 各模式 + 两版编译 + SASS） | `tmp/outputs/D11/` |
| 2026-10-12 | racecheck `counter-shared`（16 errors）+ initcheck `uninit`（4 errors） | `tmp/outputs/sanitizer-2026-10-12/d11-*` |
