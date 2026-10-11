# EXP-0008 — 浮点与数值语义（FMA 收缩 / FTZ / fast math / 求和可复现性）

## 状态

done（2026-10-08）

## 问题

1. 默认 nvcc 下 `a*b+c` 是不是 IEEE 的两步（先乘后加）语义？`--fmad` 默认是什么？
2. `--ftz=true` / `-use_fast_math` 到底改了什么？denormal 结果/输入会怎样？
3. `-use_fast_math` 是否改变 `/`、`sqrtf`、`sinf` 的指令与 bit？
4. `atomicAdd(float)` 的求和结果是否可复现？其 denormal 行为与地址空间有关吗？

## 假设

- H1：`a*b+c` 默认被收缩为 `fma.rn.f32`；`-fmad=false` 才得到两步结果。
- H2：`--ftz=true` 把 subnormal 结果与输入都冲零，`x == 0` 会翻转。
- H3：`-use_fast_math` → `div.approx` / `sqrt.approx` / `sin.approx`，且给 `__fdiv_rn` 也加 `.ftz`。
- H4：`atomicAdd(float)` 求和 bit 不可复现；global 永远 FTZ、shared 支持 denormal。

## 方法

一个 `scripts/fp_demo.cu`，四个模式（`fp` / `sum` / `atomden` / `probe`），
用 5 组编译选项构建（`default` / `-fmad=false` / `-use_fast_math` / `--ftz=true` / `-G`），
`scripts/run.ps1` 一次跑完并导出 PTX / SASS。

| 模式 | 做什么 | 观察 |
|---|---|---|
| `fp` | 一组浮点表达式，打印结果 bit | FMA 收缩、FTZ、div/sqrt/sin 差异 |
| `sum` | 1M 随机 float，线程内求和 + `atomicAdd`，跑 5 次 | 求和 bit 是否复现 |
| `atomden` | `atomicAdd(0, 1e-40f)` 到 global vs shared | 原子 denormal 语义 |
| `probe` | 一批 q 的 `1/q`、`sqrtf(q)` | fast math 的逐值 bit 差异 |

编译：`vcvars64.bat` → `nvcc -O3 -lineinfo -arch=sm_89`。

## 结果

产物：`../../tmp/outputs/D08/2026-10-08/`。

- **FMA 收缩**：默认 `a*b+c = 0x33800000`（`2^-24`）；`-fmad=false` / `-G` = `0x00000000` → 支持 H1。
- **FTZ**：`m*m` 从 `0x000116c2`（subnormal）变 `0x00000000`，`== 0` 变 true → 支持 H2。
- **fast math**：PTX 出现 `div.approx.ftz.f32` / `sqrt.approx.ftz.f32` / `sin.approx.ftz.f32` / `mul.ftz.f32`；`__fdiv_rn` 变 `div.rn.ftz.f32`；`sinf(1)` bit 改变、`sqrtf` 多个输入末位差 1 → 支持 H3。
- **求和**：5 次出现 3 个不同 bit，且都不同于 host 顺序求和 → 支持 H4。
- **原子 denormal**：global = `0x00000000`，shared = `0x000116c2` → 支持 H4。

## 结论

- 默认 nvcc **允许 FMA 收缩**（`--fmad=true`），`a*b+c` 默认不是 IEEE 两步语义；要逐操作 IEEE 用 `-fmad=false` 或显式 intrinsic。
- `--ftz=true` / `-use_fast_math` 会静默把 subnormal 结果与输入冲零，并改变 `/`、`sqrtf`、`sinf` 的指令与精度；`-use_fast_math = --ftz=true --prec-div=false --prec-sqrt=false --fmad=true`。
- `atomicAdd(float)` 的求和顺序不保证 bit 复现；其 denormal 行为由**地址空间**决定（global 永远 FTZ，shared 支持 denormal），与 `-ftz` 无关。

## 未决

- 只测 float；double / half / bf16 / fp8 / tensor core 未跑。
- `atomicAdd` 的 SASS（`RED`/`ATOM` 的 FTZ 修饰）未看。
- `-use_fast_math` 是否触发更激进的重结合/向量化未确认。
- denormal 的性能代价未量化。

## 运行记录

| 时间 | 命令 | 产物 |
|---|---|---|
| 2026-10-08 | `pwsh -File scripts/run.ps1` | `../../tmp/outputs/D08/2026-10-08/` |
