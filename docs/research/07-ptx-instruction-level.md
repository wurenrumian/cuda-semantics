---
title: 指令级 / PTX 语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-11
tags: [cuda, semantics, ptx, sass, instruction-level, cache-operator, ldg, restrict, sm_89]
related:
  - INTENT.md
  - docs/plan/cuda-semantics-topic-map.md
---

# 域 07 — 指令级 / PTX 语义

> 目标机：**sm_89 (Ada, RTX 4060 Laptop) / CUDA 12.8**。
> 本文只回答一个问题：**PTX/SASS 指令层到底向程序员承诺了什么、不承诺什么。**
> 所有关键论断都配了 `tmp/outputs/D07/2026-10-08/` 下的最小实测。

## 一句话

PTX 的访存指令把"访存"拆成三个**正交**旋钮——**序（order）× 作用域（scope）× 缓存提示（cache operator）**；
其中 cache operator 只是**性能提示**，不改变一致性语义；而 `__ldg` / `const __restrict__` 是给编译器的**承诺**，
不是硬件保证——**撒谎即 UB，而且这种 UB 在 SASS 里看得见、在运行结果里摸得着**。

## 承诺与边界

### 1. 指令层的三组旋钮

| 维度 | PTX 写法 | 承诺 | 不承诺 |
|---|---|---|---|
| **弱访存** | `ld.global` / `st.global` | 读到某条一致的值 | **不做任何顺序/可见性承诺**；可被编译器重排、合并、删除 |
| **缓存提示** | `ld.{ca,cg,cs,lu,cv}` / `st.{wb,cg,cs,wt}` | 只是给硬件/编译器的 **performance hint** | **不改变 memory consistency**(PTX ISA §9.7.9.1 原文) |
| **强访存** | `ld.acquire.gpu` / `st.release.gpu`、`.relaxed`、`.sc` + scope `.cta/.gpu/.sys`、`fence`/`membar` | 建立 `happens-before` / 原子性 | 并不比显式写的更强；scope 写小了就只在该 scope 内有效 |
| **volatile** | `ld.volatile` / `st.volatile` | 等价于 **system-scope 的 relaxed 操作** + 额外约束（§8.4.2） | **不是 acquire/release**，不建立同步；只阻止编译器把它删掉/合并 |
| **只读路径** | `ld.global.nc`（`__ldg`、`const __restrict__`） | 编译器可假设"不写、可缓存、可外提" | **没有硬件强制**；一旦真被写，读到旧值是"符合契约"的行为（UB） |

关键区分：**cache operator 管"放不放缓存 / 放哪一级"**，**memory order 管"先后可见性"**。
两者在不同输入下可以产生**相同的 SASS 助记符**（见反例 3），所以**不能从 SASS 反推语义**。

### 2. sm_89 上的实测映射（PTX → SASS）

用 `nvcc -arch=sm_89 -ptx` / `-cubin` + `cuobjdump -sass` 得到（见 `tmp/outputs/D07/2026-10-08/06_ptx_cache.*`）：

| PTX | SASS (sm_89) |
|---|---|
| `ld.global`（默认） | `LDG.E` |
| `ld.global.ca` | `LDG.E.STRONG.SM` |
| `ld.global.cg` | `LDG.E.STRONG.GPU` |
| `ld.global.cs` | `LDG.E.EF` |
| `ld.global.lu` | `LDG.E.LU` |
| `ld.global.cv` | `LDG.E.STRONG.SYS` |
| `ld.volatile.global` | `LDG.E.STRONG.SYS` |
| `ld.acquire.gpu.global` | `LDG.E.STRONG.GPU` |
| `ld.global.nc`（`__ldg`） | `LDG.E.CONSTANT` |
| `st.global.wb` / `.cg` / `.cs` / `.wt` | `STG.E.STRONG.SM` / `..STRONG.GPU` / `..EF` / `..STRONG.SYS` |
| `st.release.gpu` | `STG.E.STRONG.GPU` |

两个反直觉点：
- **默认 `ld.global` 与显式 `ld.global.ca` 的 SASS 不同**（`LDG.E` vs `LDG.E.STRONG.SM`），尽管 `.ca` 文档上就是默认缓存策略。说明 SASS 后缀编码的**不止缓存策略**。
- **`ld.global.lu` 在全局地址上按文档应表现为 streaming（≈ `.cs`）**，但 SASS 给的是 `LDG.E.LU`（而 `.cs` 给 `LDG.E.EF`）。PTX ISA 的原话是："The `ld.lu` instruction performs a load cached streaming operation (`ld.cs`) on global addresses."（§9.7.9.1）——即**文档说二者等价，SASS 却给不同后缀**。这说明 SASS 后缀并不是 PTX 语义的 1:1 翻译（见「未决」4）。

### 3. 边界（哪些不能依赖）

- `__restrict__` / `const` / `__ldg` 是**对编译器的承诺**（CUDA C++ Guide §7.2.6 / §7.10）。违反 = UB；编译器**有权**把 load 外提、CSE、甚至删除——运行期表现为"读到旧值"，而不是报错。
- cache operator 不会让脏数据变干净。`.ca` 明确写着："多个 L1 对 global 数据**不**一致；可能读到 stale L1 数据"（PTX ISA §9.7.9.1）。
- `volatile` 只保**指令数**不被删/合并，**不建立 acquire/release 同步**——它本质上是 **system-scope 的 relaxed 操作**，跨线程不带 happens-before（PTX ISA §8.4.2）。
- 无同步的全局访存，编译器可以自由重排/消除——"能通过 SASS 看到它被删了/挪走了"正是本文反例的来源。

## 反例（想当然会错）

### 反例 1（头牌，运行期可复现）：违背 `const __restrict__`，第二次读被 CSE 成旧值

"读-写-再读，第二次一定读到新值" —— 错。给编译器 `__restrict__` 承诺后，它会认为对 `q` 的写
**不可能影响 `p`**，于是把第二次 `p[0]` 直接复用成第一次的寄存器值。

`experiments/EXP-0007-ptx-instruction-level/scripts/04_restrict_alias.cu`，令 `q` 与 `p` 实际指向同一块内存：

```cpp
__global__ void alias_break(float* q, const float* __restrict__ p, float* out) {
    float before = p[0];   // 加载 #1
    *q = 5.0f;             // 通过 q 写（实际别名到 p）
    float after = p[0];    // 加载 #2：编译器认为与 *q 无关 → 复用 before
    out[0] = before; out[1] = after;
}
```

实测（`-arch=sm_89 -O3`）：

```
[restrict] before=1.000000 after=1.000000 (buf now=5.000000)   ← after 仍是旧值 1
[plain   ] before=1.000000 after=5.000000 (buf now=5.000000)   ← 去掉 restrict 才正确
```

SASS 铁证：`alias_break` 全程只有**一条** load，夹着一个 store：

```
/*0040*/  LDG.E.CONSTANT R3, [R2.64] ;   <- 唯一一次加载
/*00a0*/  STG.E [R4.64], R9 ;            <- 写 5.0 到 q
/*00b0*/  STG.E [R6.64], R3 ;            <- out[0] = 旧值
/*00c0*/  STG.E [R6.64+0x4], R3 ;        <- out[1] = 旧值（复用，没有第二次加载）
```

`alias_ok`（无 `restrict`）则有**两条** load，中间隔着 STG。PTX 层同样可见：`alias_break`
只有一条 `ld.global.nc.f32`，`alias_ok` 有两条 `ld.global.f32`。

> 这不是"硬件 bug"，而是**数据竞争 + 违背 restrict 的 UB**。重点是：**你以为在测硬件，其实在测编译器。**

### 反例 2：`__ldg` 不是"另一条只读指令"

"`__ldg` 会走特殊只读缓存路径，和普通 `p[i]` 不一样" —— 在 sm_89 上**两者逐字节相同**：

```
via_ldg   : LDG.E.CONSTANT R3, [R2.64] ;  /* 0x0000000402037981 */ /* 0x000ea2000c1e9900 */
via_plain : LDG.E.CONSTANT R3, [R2.64] ;  /* 0x0000000402037981 */ /* 0x000ea2000c1e9900 */
```

`__ldg(&p[x])`（`03_ldg_vs_plain.cu::via_ldg`）与 `const __restrict__ float* p` 的普通加载
（`via_plain`）产生**完全一致**的 SASS（连控制位都一样）。对编译器而言 `__ldg` 只是显式写出
"这是只读路径"（PTX `ld.global.nc`）的一种方式；真正决定走 `.nc/.CONSTANT` 路径的是**只读承诺**，
不是内建函数本身。

### 反例 3：不能凭 SASS 的 `STRONG.SCOPE` 后缀判断内存序

`ld.global.cg`（**弱访存**，cache op 仅绕过 L1）和 `ld.acquire.gpu`（**强访存**，建立
`happens-before`）在 sm_89 上**都**显示为 `LDG.E.STRONG.GPU`：

```
ld.global.cg    ->  LDG.E.STRONG.GPU  (01_cache_ops.sass / 06_ptx_cache.sass)
ld.acquire.gpu  ->  LDG.E.STRONG.GPU  (02_strong_vs_weak.sass, _Z10strongld_kPKiPi)
```

两者语义天差地别，SASS 助记符却相同（第二控制字不同，但不看控制位就无法区分）。
**结论：读 SASS 判断同步正确性是不可靠的；要回到 PTX 源或控制位。**

### 反例 4：无用的 load 会被整条删除，`__restrict__` 下循环不变量会被外提

- `dead_load`：`float unused = p[0];` 的结果没人用 → **SASS 里根本没有 `LDG`**，只剩 `STG`。
- `hoist`：`for (i<n) acc += p[0];` 中 `p` 为 `const __restrict__` → `p[0]` 被提到循环外，
  整个循环只有**一条** `LDG.E.CONSTANT`，循环体全是 `FADD`。

对"访存是可见副作用"的直觉，这两点都会打脸（详见 `03_ldg_vs_plain.sass`）。

## 本机验证（命令 + 观察）

### 环境与命令

```powershell
# 1) 先进入 MSVC 环境
call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
# 2) 编译（-arch=sm_89）
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -arch=sm_89 -O3 -lineinfo x.cu -o x.exe
# 3) 取 PTX / SASS
nvcc -arch=sm_89 -O3 -ptx   x.cu -o x.ptx
nvcc -arch=sm_89 -O3 -cubin x.cu -o x.cubin
cuobjdump -sass x.cubin > x.sass
```

一键复现：`experiments/EXP-0007-ptx-instruction-level/scripts/run.ps1`（编译 6 个源文件并运行）。

### 观察清单

源码在 `experiments/EXP-0007-ptx-instruction-level/scripts/`；下面的 `.ptx` / `.sass` 产物在 `tmp/outputs/D07/2026-10-08/`。

| # | 文件 | 观察 |
|---|---|---|
| 01 | `01_cache_ops.cu/.ptx/.sass` | PTX 出现 `ld.global.{ca,cg,cs,cv,lu}`、`st.global.{wb,cg,cs,wt}`；SASS 映射见上表 |
| 02 | `02_strong_vs_weak.cu/.ptx/.sass` | `ld.acquire.gpu` / `st.release.gpu` / `ld.volatile` / `cuda::atomic_ref` (acquire/release) / `seq_cst` 各自的 PTX 与 SASS |
| 03 | `03_ldg_vs_plain.cu/.sass` | `__ldg` ≡ `const __restrict__` 普通加载；`dead_load` 无 LDG；`hoist` 只有 1 条 LDG |
| 04 | `04_restrict_alias.cu/.sass` | **反例 1**：restrict 版 after=旧值，普通版 after=新值 |
| 05 | `05_inline_ptx.cu/.sass` | inline PTX：`%laneid`→`S2R R5, SR_LANEID`，`%clock64`→`CS2R SR_CLOCKLO`，`bar.sync`→`BAR.SYNC`（本机为 `BAR.SYNC.DEFER_BLOCKING`） |
| 06 | `06_ptx_cache.cu/.sass` | 直接写 PTX cache op，排除 intrinsic 干扰，得到上面的映射表 |

### 关键运行输出

```
01_cache_ops    : cudaGetLastError=no error out[0]=1.000000
02_strong_vs_weak: cudaGetLastError=no error out=7
03_ldg_vs_plain : cudaGetLastError=no error out[0]=1000000.000000
04_restrict_alias: [restrict] before=1 after=1 (buf=5) ; [plain] before=1 after=5 (buf=5)
06_ptx_cache    : cudaGetLastError=no error out=7
```

### 顺带观察到的两件事

- **`seq_cst` 更贵，SASS 里看得见**：`cuda::atomic_ref<..., system>.load(seq_cst)` 生成
  `MEMBAR.SC.SYS` + `LD.E.STRONG.SYS` + `CCTL.IVALL`；而 `acquire/release (gpu)` 是 `LD.E.STRONG.GPU`
  / `MEMBAR.ALL.GPU` / `ST.E.STRONG.GPU`。（`02_strong_vs_weak.sass`）
- **inline PTX 的 `memory` clobber 不能省**：`05` 里的 `fake_fence` 故意用
  `asm volatile("bar.sync 0;" ::: )`（缺 `"memory"`）冒充屏障，这是错的。`bar.sync` 在硬件层面
  确实是 CTA 内的屏障，但内联汇编没有 `"memory"` clobber 时，编译器**不知道**这条汇编会读写内存，
  于是可以把它前后的访存重排、合并甚至删掉——屏障语义被编译器直接绕过。正确写法要带 `: "memory"`，
  或干脆用 `__syncthreads()`。
  这里要如实说明的是：在这个最小例子里（单线程，且三个指针都是普通 `int*`，编译器必须假设它们可能
  互为别名），实测 SASS 恰好是"存 `*data` → `BAR.SYNC` → 读 `*flag`"，**并没有真的乱序**。所以这里的
  错属于**契约层面**——代码**允许**编译器乱序，只是这次没触发；换成不别名、或只读的 `__restrict__`
  指针，乱序就可能真的出现。

## 出处

**CUDA C++ Programming Guide 12.8**（下列是**在线 HTML 版**的章节号；**PDF 版**编号不同，例如本节的 §7.10 `__ldg` 在 PDF 里是 §10.10）：
- §7.10 Read-Only Data Cache Load Function（`__ldg`）
- §7.11 Load Functions Using Cache Hints（`__ldcg/__ldca/__ldcs/__ldlu/__ldcv`）
- §7.12 Store Functions Using Cache Hints（`__stwb/__stcg/__stcs/__stwt`）
- §7.2.6 `__restrict__`
- §7.5 Memory Fence Functions
- §7.1.4 Undefined behavior（`__restrict__` 违背即 UB）

**PTX ISA 8.7**（随 CUDA 12.8 发布；章节号以 8.7 为准）：
- §9.7.9.1 Cache Operators —— 原文："Cache operators on load or store instructions are treated as
  **performance hints only**. The use of a cache operator on an `ld` or `st` instruction does **not
  change the memory consistency behavior** of the program." 并给出 `.ca` 的 L1 不一致说明。
- §9.7.9.2 Cache Eviction Priority Hints
- §9.7.9.8 `ld`、§9.7.9.9 `ld.global.nc`、§9.7.9.11 `st`
- §9.7.13.4 `membar`/`fence`
- §8.4.2 volatile Operation（volatile = system-scope relaxed + 额外约束；保指令数不保操作数；
  建议改用 `ld.relaxed.sys`）
- §8.5 Scope、§8.7 Morally strong operations、§8.8 Release and Acquire Patterns、§8.9 Ordering of memory operations

**inline PTX**：*Inline PTX Assembly in CUDA*（操作数约束 `"r"/"l"/"f"/"=r"/"+r"` 与 `"memory"` clobber 的官方说明）；
操作数类型另见 PTX ISA §6 Instruction Operands。

**目标架构**：`sm_89` 相关约束见 CUDA C++ Programming Guide §6 CUDA-Enabled GPUs 与 §16 Compute Capabilities。

## 未决

1. **SASS `STRONG.SM/GPU/SYS` 的精确语义未官方确认**。SASS 无公开 ISA 手册。目前只能推断它同时编码
   "缓存解析层级"（SM/L1、GPU/L2、SYS/DRAM）与某种序属性；`.ca→STRONG.SM`、`.cg→STRONG.GPU`、
   `.cv→STRONG.SYS` 与"解析层级"高度吻合，但**待用 `nvdisasm` 控制位或更多样本证实**。
2. **compute-sanitizer 本机无法附着**：对任何 CUDA 程序（含 `cudaMalloc/cudaFree` 最小例）都报
   `Error: Target application terminated before first instrumented API call`（2025.1.0，Windows）。
   因此反例 1 的竞争**没能**被 sanitizer 自动判定，只能靠 SASS/PTX 人工判读。需排查（杀软注入/权限/PATH）。
3. **cache operator 的实际缓存行为未用 `ncu` 验证**：`.cg` 是否真的绕过 L1、`.cs` 的 evict-first 命中率，
   需 L1/TEX 指标佐证（本轮只看了指令编码）。
4. **`ld.global.lu` 在 SASS 为 `LDG.E.LU`**，与 PTX 文档"全局地址上等价于 `.cs`"的说法表面不符——
   待确认是文档措辞、目标架构差异，还是 ptxas 保留 `.LU` 语义。
5. 与相邻域的边界：本文只到"**指令提供了哪些旋钮**"；**序的推理（happens-before、scope、fence 组合）**
   归域 04/05，**编译器重排与 UB** 的更多形态归域 11。本文的 `volatile`/`seq_cst` 观察与它们有意交叉，未去重。
6. 未展开：`.mmio`、`ld.relaxed`、`.el/.evict_*` 优先级提示、`cp.async` 路径（归域 09）。
