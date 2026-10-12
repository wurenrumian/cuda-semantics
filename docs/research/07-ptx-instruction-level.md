---
title: 指令级 / PTX 语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-12
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

## 四层链路：谁承诺、谁观察

读任何指令级讨论之前，先把这条链钉在脑子里——**不同层出的字据不一样**：

```
  源码 (CUDA C++)                ── 语言模型层：语义承诺的唯一出处(规范原文)
      │ nvcc 前端
      ▼
  PTX (虚拟 ISA)                 ── 中间契约层：可全局通吃各代 GPU(JIT/向前兼容,域 12);
      │ ptxas                        order/scope/cache-op 在这层还有名字,可读性最好
      ▼
  SASS (目标机器码)              ── 实现层：dm 无公开手册;**只能当证据,当不了定义**
      │ 装载
      ▼
  硬件执行                       ── 物理层：时序、发射、缓存行为(ncu 才看得到,权在域 12)
```

- **PTX 能证明什么**：程序"被允诺"的语义动了哪些旋钮（`ld.acquire.gpu` 就是 acquire），指令
  有没有被编译器保留——因为 PTX 在语义层面仍有正式定义（PTX ISA）。
- **PTX 不能证明什么**：最终硬件上有没有、几条、怎么发射——它会被 ptxas 进一步变换。
- **SASS 能证明什么**："这份编译器在这块芯片上实际生成了什么指令"——删了、外提了、加了几条
  MEMBAR,这类**实现事实**只有它有。
- **SASS 不能证明什么**：语义承诺本身。同一个 SASS 助记符可以覆盖多种 PTX 语义（反例 3）、
  同一 PTX 也可因架构落成不同 SASS——**从 SASS 反推"应该这样写"是把证据当条文**。
  正确性论证永远回语言模型层(域 04/05/11),SASS 只是验证"这台机器上语义被落成了什么"。

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

三个旋钮**真的互相独立**——读任何一条指令按"各自有没有设"分别打勾：

- `ld.global.cg`：弱访存 + 缓存绕 L1。**没有任何顺序承诺**（加了缓存提示 ≠ 加了序）。
- `ld.acquire.gpu.global`：acquire 序 + 普通缓存路径。换 cache op（`.cg`/`.ca`）不改变它的
  acquire 语义，只换缓存行为。
- 同一条 PTX 上两者可并存：`ld.acquire.gpu.global.cg`——序归序、缓存归缓存，各自设各自生效。
- `volatile` 则是三件的第四种配法：**system-scope relaxed + "不许删/不许合并"指令约束**，没有
  acquire/release，也没有缓存语义上的承诺（域 04/05 已实测它编译成 `LDG.E.STRONG.SYS`，与
  `.cv` 同形——形同而语义不同，正因如此才说"不能从指令形反推语义"）。

成对实验法（本文全部反例的方法论）：**每次只动一个旋钮**，其余保持不变，看 PTX/SASS 落点
动不动。两个旋钮一起动，就再也分不清哪个旋钮起了作用。

### 2. sm_89 上的实测映射（PTX → SASS）

> **标签**：本表是**本机 sm_89 + CUDA 12.8 特定命令**（`nvcc -arch=sm_89 -ptx/-cubin` +
> `cuobjdump -sass`，见 §环境与命令）的实现证据，**观察范围仅限这些产物**——换 arch、优化
> 选项或工具版本都可能变。按链路图的定位，它是"这份编译器怎么兑现"，**不是**语义条文。

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

> 这不是"硬件 bug"，而是**数据竞争 + 违背 restrict 的 UB**。也要说清责任链：`const __restrict__`
> 是**调用方-被调方之间的契约**（调用方承诺 p 不会被别的指针在某段时间内写），出了别名就是**调用方
> 违约**；编译器随后做的一切（复用寄存器、删第二次 load）都是**规范许可的合法优化**，不是优化器
> "随机重排"、更不是"GPU 缓存不刷新不失效"的硬件证据——SASS 里那条唯一的 `LDG.E.CONSTANT` 就是
> 契约被合法消费的现场。**你以为在测硬件，其实在测编译器（怎么兑现代理给你的承诺）。**

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

三点适用边界（写代码时定心用）：

1. **前提是"真只读"**：这段数据在整个 kernel 运行期间**不被任何人写**（同 kernel 内的别名写、
   其他 kernel/线程的并发写、映射到 host 侧的写都算）。前提破 → 读到旧值是"契约内"的行为（UB）。
2. **通路是编译器选的，不是你可以指定的硬件开关**：即便写了 `__ldg`，若编译器判定不合算/不安全
   （地址本地性、寄存器压力、后续有写可能），它可以**不**走 `.nc`；反过来，`const __restrict__`
   的普通 load 编译器也可能**自作主张**用 `.nc`。所以"我用了 `__ldg` 就一定走 texture 路径"
   是逆向的想当然——判据只有 PTX/SASS 产物。
3. **别处的架构上，`__ldg` 与普通 load 可能落到不同通路甚至被合并消除**：sm_89 的
   `LDG.E.CONSTANT` 与普通 `LDG.E` 同族；老架构（Kepler 时代）`__ldg` 曾真的对应独立 texture
   单元路径，未来也可能改。**"本机逐字相同"是本机事实，不是跨代承诺。**

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

### 怎样读反汇编——本文实验同款的最短指南

本文所有"SASS 铁证"都按这五步得出，复现/质疑时也照五步走：

1. **定位 kernel**：`cuobjdump -sass <cubin/exe>` 输出按 `Function : _Z…` 分段；名字是
   C++ mangle 过的（`_Z` 开头）。实在混淆时用 `nvcc --keep`，或给 kernel 起独占名字。
2. **对回源码行**：编译时带 `-lineinfo`（本文实验均带），`nvdisasm -c` 可见行号归属；
   定不到行号的（编译器生成的胶水）别硬安给某一行。链接到对应源码是引用机器证据的第一步。
3. **追谓词与分支**：`@P0 BRA`、`@!P0 LDS`——**被谓词保护的指令是"会执行的路径"**，按谓词
   真假展开读；`BRA.CONV` 是收敛相关分支（域 01）。屏障/栅栏要确认在一条**可达**路径上。
4. **分清"被删的代码"与"未执行的代码"**：SASS 里**根本不存在** ⇒ 编译器删了/合并了
   （反例 4 的 dead_load）；存在但谓词恒假/不可达 ⇒ 死路径。两种"消失"原因不同——
   一个是编译器许可，一个是运行时事实，引用时别混。
5. **核不出语义就声明未定**：同助记符不同语义时（`STRONG.SYS` 双义、`.LU/.EF` 差异），
   用 `nvdisasm -c` 看控制位；推不下去就写进「未决」，不要现场编语义。

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
2. **反例 1 无运行时工具判定（环境限制，如实标注）**：全局内存上的"丢失第二批 load"不属
   racecheck 视野（racecheck 只查 shared memory hazard，04/05 已实测校准）；memcheck 也不管
   "语义读旧值"。2026-10-11 起 compute-sanitizer 本机已可用（根因 Windhawk 注入冲突，见 00 篇），
   但对本反例仍无对应工具——结论**只能**靠 SASS/PTX 人工判读，不能用"跑不出错"补位。
3. **cache operator 的实际缓存行为未用 `ncu` 验证**：`.cg` 是否真的绕过 L1、`.cs` 的 evict-first 命中率，
   需 L1/TEX 指标佐证（本轮只看了指令编码）。
4. **`ld.global.lu` 在 SASS 为 `LDG.E.LU`**，与 PTX 文档"全局地址上等价于 `.cs`"的说法表面不符——
   待确认是文档措辞、目标架构差异，还是 ptxas 保留 `.LU` 语义。
5. 与相邻域的边界：本文只到"**指令提供了哪些旋钮**"；**序的推理（happens-before、scope、fence 组合）**
   归域 04/05，**编译器重排与 UB** 的更多形态归域 11。本文的 `volatile`/`seq_cst` 观察与它们有意交叉，未去重。
6. 未展开：`.mmio`、`ld.relaxed`、`.el/.evict_*` 优先级提示、`cp.async` 路径（归域 09）。
