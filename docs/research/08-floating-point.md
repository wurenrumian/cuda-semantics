---
title: 域 08 — 浮点与数值语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-12
tags: [cuda, semantics, floating-point, ieee754, fma, ftz, fast-math, denormal, sm_89]
related:
  - EXP-0008
  - docs/plan/cuda-semantics-topic-map.md
---

# 域 08 — 浮点与数值语义（sm_89 / CUDA 12.8）

## 一句话

CUDA 的浮点**默认**对单精度基本运算保持 IEEE-754 语义（正确舍入、保留 denormal），但**默认允许 FMA 收缩**，并且 `-use_fast_math` / `--ftz` / `--prec-div` / `--prec-sqrt` 会逐项放弃这些保证；所以"结果是否 bit 可复现"取决于**编译选项 + 求和顺序 + 地址空间**，而不是硬件本身。

## 概念地图：影响浮点结果的六个旋钮

同一行 `x = a*b + c` 的最终 bit，由**六个互不替代的东西**共同决定——诊断任何"数值不对"时先想这张图：

| 旋钮 | 改bit的情形 | 本篇哪里讲 | 谁管 |
|---|---|---|---|
| ① IEEE-754 基本运算 | 每步都正确舍入（±0 ULP），语义层承诺 | 承诺 | 规范本身（原子路径有例外，见 ③） |
| ② FMA contraction | `a*b+c` 少一次舍入 → 与逐步算不同 | 反例 1 | 编译器（`--fmad`，默认开） |
| ③ FTZ/denormal | subnormal（输入、中间值或输出）被冲成 ±0 | 反例 2/3 | 编译选项 `--ftz`；**原子的地址空间独立硬规定** |
| ④ 近似数学函数 | `sinf/sqrtf/div` 从"正确舍入"换"几 ULP 近似" | 反例 4 | `-use_fast_math` 的 flag 集合 |
| ⑤ 并行归约顺序 | 浮点加不满足结合律 → 分块/调度不同，低位浮动 | 反例 5 | 算法与调度（**可复现性需求**，需自己固化） |
| ⑥ 原子到达顺序 | `atomicAdd(float/double)` 的累加次序不可控 | 反例 5 | 调度（同 ⑤；跨域 04/05） |

读法提醒：**⑤⑥ 与 ①②③④ 是两类问题**——⑤⑥ 的"每次结果不同"是**合法的**（每一步
单独看都是没毛病的浮点加，只是"哪些步先做"变了）；①–④ 是"单条指令的语义被改变"。
混淆这两类是浮点调试里最常见的岔路（"是不是显卡算错了？"几乎都出自这里）。

## 承诺与边界

### 承诺（默认 `-O3`，无 `--use_fast_math`）

- `x+y`、`x-y`、`x*y`、`x/y`、`1/x`、`sqrtf(x)` 对 `float`/`double` 遵守 IEEE-754，round-to-nearest-even，**最大 0 ULP**（正确舍入）。
- **denormal（subnormal）默认被支持**：单精度运算保留 subnormal 输入与结果（sm_20 及以后）。见 PTX ISA §9.7.3 原文："Single-precision instructions support subnormal inputs and results by default for `sm_20` and subsequent targets."
- 显式 intrinsic `__fadd_rn` / `__fmul_rn` / `__fmaf_rn` / `__fdiv_rn` / `__fsqrt_rn` 提供**指定舍入模式**的逐操作版本，并保证"**不会被合并成 FMAD**"。

### 边界（不承诺 / 会变）

- **FMA 收缩默认开启**（`--fmad=true` 是 nvcc 默认）。`a*b+c` 这样的表达式"**通常**"被合成一条 `fma.rn.f32`，只舍入一次；它不是 `round(round(a*b)+c)`。PTX ISA 明确允许：`{add,sub,mul}.rnd.f32` "If no rounding modifier is specified, default is `.rn` and instructions **may be folded into a multiply-add**." 想要逐操作 IEEE 语义必须 `-fmad=false` 或写显式 intrinsic。
- **FTZ 是全局开关**：`--ftz=true`（`-use_fast_math` 会带上）把单精度运算的 **subnormal 输入和结果都冲成符号保留的零**（PTX `.ftz` 修饰符）。它不只会冲掉"极小输入"，连**正常输入相乘得到的 subnormal 结果**也会被变成 0。
- **`-use_fast_math` 是一整套语义让步**，等价于 `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`，并额外把一批数学库函数替换成 intrinsic：
  - `--prec-div=false`：`x/y` 走近似除法（`div.approx.f32`，几 ULP），`x/y` 也被映射为 `__fdividef`。
  - `--prec-sqrt=false`：`sqrtf(x)` 走 `sqrt.approx.f32`（几 ULP；注意 `sqrtf` **不在** guide 的 `-use_fast_math` 函数替换表里，是这条 flag 在起作用）。
  - 函数替换（guide §13.2 的 *Functions Affected by -use_fast_math* 表）：`sinf/cosf/tanf/sincosf`、`logf/log2f/log10f`、`expf/exp10f`、`powf`、`tanhf` → 对应 `__*` intrinsic（精度下降 + **特殊值处理不同**）。
- **连显式 intrinsic 也不豁免 FTZ**：`-use_fast_math` 下 `__fdiv_rn(p,q)` 编译成 `div.rn.ftz.f32`（舍入仍是正确舍入，但加了 `.ftz`）。所以 `__f*_rn` 保证的是"**不收缩 + 指定舍入**"，不是"**不受编译选项影响**"。
- **原子浮点的 denormal 行为由地址空间决定，与 `-ftz` 无关**（文档硬规定）：`atomicAdd(float)` 落在 **global** 时**永远** FTZ（等价 `add.rn.ftz.f32`）；落在 **shared** 时**支持 denormal**（等价 `add.rn.f32`）。同一次 `atomicAdd` 调用，换个地址空间就换语义。
- **可复现性没有承诺**：`atomicAdd(float)` 的到达顺序由调度决定；不同编译选项、不同 `-arch`、不同求和顺序都可能给出不同 bit。CUDA 只承诺"在某一次具体执行里，每条指令的语义"，不承诺跨运行/跨编译的 bit 一致。

### 与相邻域的关系

- FMA/FTZ 影响的是**指令语义**，观察窗口是 PTX/SASS（域 7）。
- `atomicAdd(float)` 的顺序问题属于**原子 + 内存序**（域 4/5）。
- 半精度 / bf16 / fp8 / tensor core 的 fragment 布局属**架构语义**（域 12），本篇只提边界，不在本机跑 mma。

## 反例

### 反例 1（主）：`a*b+c` 默认**不是**"先乘后加"

想当然：IEEE-754 下 `a*b+c` = `round(round(a*b)+c)`，两次正确舍入。
实测：默认编译下它被收缩成一次舍入的 FMA。

取 `a = b = 1 + 2^-12`（`0x3F800800`），`c = -(1 + 2^-11)`（`0xBF801000`）：

| 编译 | `a*b+c` 结果 | bit |
|---|---|---|
| 默认 `-O3` | 5.96046448e-08（`2^-24`，精确融合值） | `33800000` |
| `-fmad=false` | 0（两步入 IEEE 的值） | `00000000` |
| `-G`（debug） | 0 | `00000000` |

同一份源码、同一台机器、同一架构，只因为 `--fmad` 默认开，结果从 `2^-24` 变成 `0`。**默认不是 IEEE 的两步语义。**

这笔账逐步过一遍（正是"一次舍入 vs 两次舍入"的差别）：

```
输入：a = b = 1 + 2^-12 = 0x3F800800，c = -(1 + 2^-11) = 0xBF801000

数学(实数):   a*b + c = (1+2^-12)² − (1+2^-11) = 1 + 2^-11 + 2^-24 − 1 − 2^-11 = 2^-24(精确)

E0 两步 IEEE(两次舍入)：                       ← -fmad=false / -G 落在这条
   E1 a*b = 1+2^-11+2^-24：float 只有 24 位尾数,2^-24 无法保留 → 舍回 1（舍入#1）
   E2 E1 + c = 1 − 1 = 0（舍入#2，已无误差空间）      → 0x00000000

Ef 一次 IEEE(收缩成 FMA，舍入#1 即终)：        ← 默认 -O3 落在这条
   E3 a*b+c 三项一次做,全精度中间值直接决定 ±0.5 ULP → 保留 2^-24   → 0x33800000
```

两种都"没算错"——各自是对**自己那套语义**的正确舍入。想逐操作地拿两步舍入，只能
`-fmad=false` 或显式 `__fmaf_rn/__fadd_rn/__fmul_rn`（后者保证不被合并，见承诺段）。

### 反例 2：FTZ 冲的不只是"极小输入"

想当然：只有 1e-45 这种量级的数才受 FTZ 影响。
实测：**正常输入** `m = 1e-20f`，其乘积 `m*m ≈ 1e-40` 是 subnormal。

| 编译 | `m*m` | bit | `m*m == 0` |
|---|---|---|---|
| 默认 / `-fmad=false` / `-G` | 9.9999461e-41 | `000116c2` | false |
| `--ftz=true` 或 `-use_fast_math` | 0 | `00000000` | **true** |

所以一个"输入正常"的表达式，在 fast math 下会静默变成 0，并且 `if (x == 0.0f)` 分支翻转。

### 反例 3（最反直觉）：同一个 `atomicAdd(float)`，global 永远 FTZ，shared 支持 denormal

想当然：`atomicAdd` 的数值行为跟着 `-ftz` 编译选项走。
实测（默认编译，未加任何 FTZ 选项）：`atomicAdd(dst, 1e-40f)`，`dst` 初值 0：

| 目标地址空间 | 结果 | bit |
|---|---|---|
| global | 0（denormal 被冲掉） | `00000000` |
| shared | 9.9999461e-41（保留） | `000116c2` |

同一个 API、同一份代码、同一个编译选项，仅因地址空间不同而语义不同。文档硬规定：global 上的单精度浮点原子加**永远**等价 `add.rn.ftz.f32`，shared 上等价 `add.rn.f32`。

> **别反向推广**：这条说的是"浮点**原子**的 global 路径恒 FTZ"，**不是**"浮点运算到处恒 FTZ"。
> 普通乘/加的 FTZ 完全由 `--ftz` 控制（反例 2：默认就保留 `1e-40`）；只有原子浮点路径是
> "无论编译选项、非 FTZ 不可"。05 篇观察 6 的 SASS 对照是同一结论的指令层版本：
> `-ftz=false/true` 两版都出 `RED.E.ADD.F32.FTZ.RN.STRONG.GPU`。写"FTZ"类断言时务必带上
> **哪条指令路径**——输入、中间值、输出、指令路径四样里只有"global 原子加的路径"是恒 FTZ。

### 反例 4：`-use_fast_math` 静默换掉 `/`、`sqrtf`、`sinf` 的语义

想当然：`-use_fast_math` 只是"快一点"，结果大致一样。
实测（PTX 层）：

| 表达式 | 默认 PTX | `-use_fast_math` PTX |
|---|---|---|
| `a*b+c` | `fma.rn.f32` | `fma.rn.ftz.f32` |
| `m*m` | `mul.f32` | `mul.ftz.f32` |
| `p/q` | `div.rn.f32` | `div.approx.ftz.f32` |
| `sqrtf(q)` | `sqrt.rn.f32` | `sqrt.approx.ftz.f32` |
| `sinf(p)` | 一长串内联 `FFMA` 多项式（精确实现） | `sin.approx.ftz.f32`（`MUFU.SIN`） |
| `__fdiv_rn(p,q)` | `div.rn.f32` | `div.rn.ftz.f32`（注意多了 `.ftz`） |

可观测的 bit 差异：`sinf(1.0f)` 从 `3F576AA5` 变 `3F576AA3`；`sqrtf` 在 `q=11,19,23,37,1.5` 上末位差 1 ULP，`1/q` 在 `q=37` 上也差 1 ULP（见 `probe-*.txt`，其余输入未变）。所以 **approx 不是"总是差"，而是"不再保证正确舍入"**。

### 反例 5：同一个 kernel，同一份输入，5 次跑出不同 bit

想当然：同一个 kernel 跑两次结果 bit 一致。
实测：1M 个 `[-1,1)` 随机 float，每线程局部求和后 `atomicAdd` 到全局，默认编译跑 5 次：

```
device atomic run 0 : c30e04e0  -142.019043
device atomic run 1 : c30e04df  -142.019028
device atomic run 2 : c30e04e0  -142.019043
device atomic run 3 : c30e04de  -142.019012
device atomic run 4 : c30e04de  -142.019012
```

5 次出现 3 个不同 bit；也都不同于 host 端 float 顺序求和的 `c30dfebb`（`-141.995041`）。浮点归约的 bit 复现不是 CUDA 的承诺。

**把两类问题分开读这张表**（概念地图 ⑤⑥ vs ①–④ 的落点）：

- **每次跑结果不同**：这是 ⑤⑥（归约顺序/原子到达顺序不确定）。**每一步加法本身都是精确符合
  IEEE 的**——不许任何一步"算错"，只是"哪几步先算"由调度决定，而浮点加不结合，顺序不同
  尾位就浮动。它的对策是**需求侧**：全并确定归约（固定分组顺序/单线程累加/整数定点计数），
  或者接受误差界。
- **host 顺序和与 device 结果不同**（`c30dfebb` vs `c30e04de` 区段）：是 ⑤ 的另一个侧面——
  1M 个数分给 256 线程各局部累加，等效于换了"括号位置"；**不是**（a）硬件不符合 IEEE
  （b）原子运算本身不精确（单条 `atomicAdd` 仍是一次正确舍入的加）。差别全部来自"运算结构变了"。
- **单条指令不符合 IEEE**：那才是 ①–④ 的事（收缩/FTZ/approx），有 PTX/SASS 可查。
  三者别混着debug——查"结果不可复现"去查归约结构（或干脆换成 double/long），查"数值错了
  ULP"去查编译选项与指令。

## 本机验证（命令 + 观察）

> **测量条件标签（本篇全部 bit/数值结论共通）**：RTX 4060 Laptop（sm_89, Ada）、CUDA 12.8
> （nvcc V12.8.61）、编译与运行命令见下、`-lineinfo` 均开。bit pattern 结论都附了**可复现输入**
> （反例 1 的 `1+2^-12` 三件套、反例 2 的 `1e-20f`、反例 3 的 `1e-40f`），并有 host 端参照值
> （如反例 5 的 host double）；换编译器版本/架构需重测，时钟/温度类性能波动不影响 bit 结论
> （本篇不做计时）。

环境：RTX 4060 Laptop（sm_89, Ada），CUDA 12.8（nvcc V12.8.61）。
源码与脚本：`experiments/EXP-0008-floating-point/`（`scripts/fp_demo.cu`、`scripts/run.ps1`）。
产物：`tmp/outputs/D08/2026-10-08/`。

### 命令

```bat
call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set NVCC=E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe

%NVCC% -O3 -lineinfo -arch=sm_89                 -o fp_default.exe  fp_demo.cu
%NVCC% -O3 -lineinfo -arch=sm_89 -fmad=false     -o fp_nofmad.exe   fp_demo.cu
%NVCC% -O3 -lineinfo -arch=sm_89 -use_fast_math  -o fp_fastmath.exe fp_demo.cu
%NVCC% -O3 -lineinfo -arch=sm_89 --ftz=true      -o fp_ftz.exe      fp_demo.cu
%NVCC% -G  -lineinfo -arch=sm_89                 -o fp_debug.exe    fp_demo.cu

fp_default.exe  fp        :: 各表达式的 bit pattern
fp_nofmad.exe   fp
fp_fastmath.exe fp
fp_ftz.exe      fp
fp_debug.exe    fp
fp_default.exe  sum 5     :: atomicAdd 求和可复现性
fp_default.exe  atomden   :: global vs shared 原子 denormal
fp_default.exe  probe     :: 逐值 1/q、sqrtf
fp_fastmath.exe probe

E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe -sass fp_default.exe   > sass-default.txt
E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe -sass fp_fastmath.exe  > sass-fastmath.txt
%NVCC% -O3 -arch=sm_89 -ptx -o fp_default.ptx  fp_demo.cu
%NVCC% -O3 -arch=sm_89 -use_fast_math -ptx -o fp_fastmath.ptx fp_demo.cu
```

### 观察 1：FMA 收缩（默认 vs `-fmad=false` vs `-G`）

`fp-default.txt` / `fp-nofmad.txt`：

```
default :  contract a*b+c = 33800000 (5.96046448e-08)   mul_rn+add_rn = 00000000
nofmad  :  contract a*b+c = 00000000 (0)                mul_rn+add_rn = 00000000
debug   :  contract a*b+c = 00000000 (0)                mul_rn+add_rn = 00000000
```

PTX（默认）里 `a*b+c` 直接是 `fma.rn.f32`；`-G` 下退化为不收缩（结果同 `-fmad=false`）。

### 观察 2：FTZ（`--ftz=true` / `-use_fast_math`）

`fp-ftz.txt`、`fp-fastmath.txt`：`m*m` 与 denormal 输入 `1e-40f*2` 都从 `000116c2` / `00022d84` 变成 `0`。

### 观察 3：SASS 指令

`cuobjdump -sass` 的 `k_fp`：

| 版本 | 相关指令 |
|---|---|
| 默认 | `FFMA`、`FMUL`、`FCHK`、`MUFU.RCP` + 牛顿迭代、`MUFU.RSQ` + 牛顿迭代（正确舍入的 `/`、`sqrt`） |
| `-use_fast_math` | `FMUL.FTZ`、`FADD.FTZ`、`FFMA.FTZ`、`MUFU.SIN`、`MUFU.SQRT`；`__fdiv_rn` 仍走 `FCHK` 路径（但带 FTZ） |

### 观察 4：原子 denormal（`atomden-default.txt`）

```
atomicAdd global 0 + 1e-40f : 00000000  0
atomicAdd shared 0 + 1e-40f : 000116c2  9.9999461e-41
```

### 观察 5：求和（`sum-atomic.txt`）

见反例 5；host double = `-142.01904296875`，device 各次在 `-142.019012` ~ `-142.019043` 间浮动。

## 出处

**CUDA C++ Programming Guide（12.8）**

- §5.4.1 Arithmetic Instructions — 讨论 `-ftz=true` / `-prec-div=false` / `-prec-sqrt=false` 与性能、精度取舍。
- §13 Mathematical Functions
  - §13.1 Standard Functions — 各标准数学函数的 ULP 误差上界。
  - §13.2 Intrinsic Functions — `__fadd_[rn,rz,ru,rd]()` / `__fmul_*` "map to addition and multiplication operations that the compiler **never merges into FMADs**"；"additions and multiplications generated from the `*` and `+` operators **will frequently be combined into FMADs**"；`-use_fast_math` 语义与 **"Functions Affected by -use_fast_math" 表**；原子浮点的 denormal 行为（global 永远 FTZ、shared 支持 denormal）。

**CUDA Compiler Driver NVCC（12.8）**

- §4.2.7.7 `--use_fast_math` — "implies `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`"。
- §4.2.7.8 `--ftz` — `false` 保留 denormal；默认 `false`。
- §4.2.7.9 `--prec-div` — 默认 `true`。
- §4.2.7.10 `--prec-sqrt` — 默认 `true`。
- §4.2.7.11 `--fmad` — 默认 `true`。

**PTX ISA 8.7（随 CUDA 12.8）**

- §9.7.3 Floating-Point Instructions — 舍入修饰符 `.rn/.rz/.rm/.rp`；`.ftz` 语义："flush subnormal inputs and results to sign-preserving zero"；表注 "{add,sub,mul}.rnd.f32 … may be folded into a multiply-add"。
- §9.7.3.3 `add`、§9.7.3.5 `mul`、§9.7.3.6 `fma`、§9.7.3.8 `div`、§9.7.3.13 `rcp`、§9.7.3.15 `sqrt`、§9.7.3.18 `sin` — 各指令的语法（`.rnd`、`.ftz`、`.approx`）与语义。
- §9.7.4.4 Half Precision Floating Point Instructions: `fma` — 半精度路径的入口（本机未展开）。

## 未决

- 只系统验证了 **float**。double 的 FMA 收缩 / `-fmad` 行为、以及 `__fma_rn` 未单独测。
- **半精度 / bf16 / fp8 与 tensor core（wmma/mma）** 未在本机跑；sm_89 支持 fp8 mma，属域 12 的 fragment 布局问题，待补。
- `-use_fast_math` 是否还触发更激进的**表达式重结合 / 向量化**，未逐项确认（本篇只覆盖收缩与 approx 替换）。
- `atomicAdd(float)` 的 global-FTZ / shared-denormal 只做了**行为验证**，未看其 SASS（`RED`/`ATOM` 上的 FTZ 修饰）；`red`（不返回旧值）与 `atom` 是否同语义待查。
- denormal 在 sm_89 上的**性能代价**未量化（属"性能语义"，会随架构变，见域 12）。
- `__fdividef` / `__fsqrt_rn` 等其它 intrinsic 在 fast math 下的逐项 ULP 差异未补全。

## 运行记录

| 时间 | 命令 | 产物 |
|---|---|---|
| 2026-10-08 | `pwsh -File scripts/run.ps1`（5 变体 build + fp/sum/atomden/probe + PTX/SASS） | `tmp/outputs/D08/2026-10-08/` |
