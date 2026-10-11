# EXP-0007-ptx-instruction-level

> 一域一实验。定义与结论见 `docs/research/07-ptx-instruction-level.md`。

## 运行

```powershell
pwsh -File scripts/run.ps1
```

脚本对 `scripts/` 下每个 `.cu` 依次做三件事：编译可执行（跑起来看结果）、`-ptx`（看指令级语义）、`-cubin` + `cuobjdump -sass`（看硬件编码），最后依次运行。

产物写入 `tmp/outputs/D07/2026-10-08/`（本机，未入库）：`01..06.exe`、`01..06.ptx`、`01..06.cubin`、`01..06.sass`。

## 源文件 → 论断

| 文件 | 支持的论断 |
|---|---|
| `01_cache_ops.cu` | `__ldca/__ldcg/__ldcs/__ldcv/__ldlu` 与 `__stwb/__stcg/__stcs/__stwt` 的 PTX → SASS 映射 |
| `02_strong_vs_weak.cu` | 弱访存 vs `ld.acquire/st.release` vs `volatile` vs `cuda::atomic_ref`（acquire/release/seq_cst）各自的 SASS |
| `03_ldg_vs_plain.cu` | `__ldg` ≡ `const __restrict__` 普通加载；无用 load 被删；循环不变量被外提 |
| `04_restrict_alias.cu` | 反例：违背 `const __restrict__` 后第二次读被 CSE 成旧值（对照：去掉 restrict 才正确） |
| `05_inline_ptx.cu` | inline PTX 操作数约束；`fake_fence` 缺 `"memory"` clobber 的反面示范 |
| `06_ptx_cache.cu` | 直接用 inline PTX 写 cache operator，排除 intrinsic 包装，得到干净的映射表 |

## 环境

sm_89（RTX 4060 Laptop）/ CUDA 12.8 / nvcc V12.8.61；编译前先 `call ...\vcvars64.bat`。
