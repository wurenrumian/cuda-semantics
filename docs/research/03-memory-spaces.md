---
title: 域 03「内存空间」语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-11
tags: [cuda, semantics, memory-spaces, sm_89, local-memory, shared-memory, constant-memory, coalescing, bank-conflict]
related:
  - INTENT.md
  - docs/plan/cuda-semantics-topic-map.md
  - experiments/EXP-0003-memory-spaces/README.md
---

# 域 03 · 内存空间语义

> 环境：**sm_89（RTX 4060 Laptop, Ada）/ CUDA 12.8 / nvcc V12.8.61**。
> 实验与产物：`experiments/EXP-0003-memory-spaces/`、`tmp/outputs/D03/2026-10-08/`。

## 一句话

CUDA 有六个内存空间（**register / local / shared / global / constant / texture**），每个变量或访存都落在**恰好一个**空间里；这个空间决定了它的**作用域（谁能看见）、生命周期（活多久）、物理落点与访问代价**。而"落在哪个空间"并不完全由源码写法决定——它常常是**编译器按访问模式选的**，只能到 PTX/SASS 层面才能确认。

## 怎么读这张表：四个维度是四件独立的事

表格有四列，先别急着背内容，把四列**为什么是四列**想清楚：

- **作用域（谁能看见）**：回答"这个变量对谁可见"——只有本线程？本 block？整个 grid + host？
  这是**语义**层面的答案，写代码时就要按它设计。
- **生命周期（活多久）**：回答"它什么时候出生、什么时候死亡"。thread 级的随线程生灭，
  block 级的随 block 生灭，`cudaMalloc` 出来的活到你 `cudaFree`。
- **物理落点（住在哪）**：回答"这份数据字面上放在什么硬件里"——寄存器堆、片上 SRAM、还是 DRAM。
  这是**实现**层面的事实。关键区别：作用域/生命周期是**承诺**，物理落点常常是**编译器的选择**
  （尤其 register↔local 的分界），所以承诺和现象要分开记录。
- **访问代价（多贵）**：取决于前三者 + 访问模式（是否合并、是否 bank conflict、缓存命中）。
  它不是任何一列的直接函数，**"shared 一定比 global 快"这类结论在它面前统统作废**（见反例 3）。

而表格里"PTX state space / 典型 SASS"两列，对应的是这条编译链：

```text
源码写法            PTX（NVIR 中间层）         SASS（真机器码）           硬件资源
────────────       ──────────────────        ─────────────────────     ──────────────
__shared__ int s   .shared 声明 + ld.shared   LDS / STS                 片上 SRAM
__device__/堆      .global + ld.global        LDG / STG                 DRAM + L2(+L1)
（重索引的）数组    .local depot + ld.local    LDL / STL（或被提升→无）   DRAM（经 L1）或寄存器
                   （或被提升成寄存器）

每一步都可能"变卦"：nvcc 先把源码翻译成 PTX，ptxas 再优化成 SASS。
源码里的意图只到 PTX 为止；**最终花了多少钱，以 SASS 为准**。
这就是"落在哪个空间不完全由源码决定"的原因。
```

## 承诺与边界

### 三个完整的对照例子

**example 1 — shared**：

```cpp
__global__ void k() {
    __shared__ int s[64];          // 源码：block 级共享
    ...
    s[threadIdx.x] = v;            // 编译：PTX st.shared → SASS STS
    // 物理位置：SM 片上 SRAM；作用域：本 block 可见；生命周期：block 存续期
}
```

**example 2 — global**：

```cpp
__global__ void k(int* d) {        // d 来自 cudaMalloc（device DRAM）
    d[threadIdx.x] = v;            // 编译：PTX st.global → SASS STG
    // 物理位置：DRAM + L2（通常也过 L1）；作用域：全 grid + host；生命周期：malloc→free
}
```

**example 3 — local（最容易想错的一个）**：

```cpp
__global__ void k(int i) {
    int arr[64];                   // 源码看起来就是"数组"，没写任何 __local__
    arr[i] = v;                    // 运行时索引 → 编译器放不下寄存器堆
    // 编译：PTX .local depot + st.local → SASS STL
    // 物理位置：DRAM（跟 global 住同一块内存！），只是每线程一段私有、通常走 L1
    // 注意：如果 i 可证明在界内，ptxas 可能把它整体提升回寄存器 → SASS 里一条 STL 都没有
}
```

看到没有：local 的源码里**没有任何标记**，它的落点完全是编译器决定的。

### 四个空间的速查表

| 空间 | 作用域 | 生命周期 | 物理落点（sm_89） | PTX state space | 典型 SASS | 承诺 | 边界 / 不承诺 |
|---|---|---|---|---|---|---|---|
| **register** | 单线程 | 线程执行期 | 寄存器文件（片上） | （无地址，无 state space 名） | 直接操作数 | 最快、可定值 | 每线程数量有限（≤255）；**不可取地址**；压力大时 spill |
| **local** | 单线程 | 线程执行期 | **device memory**（与 global 同一物理内存，L1 缓存） | `.local` | `LDL` / `STL` | 每线程私有、可动态索引 | 不是片上暂存；有内存延迟；容量受栈限制；**是否真的落 local 由编译器定** |
| **shared** | 单 **block** | block 执行期 | SM 片上 SRAM | `.shared` | `LDS` / `STS` | block 内低延迟共享、片上带宽 | block 私有（跨 block 不可见）；block 起始内容**未定义**；有 bank conflict |
| **global** | 全 grid + host | `cudaMalloc`→`cudaFree` | device DRAM，经 L2 | `.global` | `LDG` / `STG` | 所有线程可见、持久 | **无自动顺序/可见性**（需同步，见域 4）；合并访问决定代价 |
| **constant** | 全 grid，**只读** | 程序期 | device memory + 常量缓存 | `.const` | `LDC` / `ULDC` | 同地址时极快（广播） | 只读（host 用 `cudaMemcpyToSymbol` 写）；**按 lane 分叉即串行化** |
| **texture / read-only** | 全 grid，只读 | 同 global | 经**只读数据缓存**（L1/TEX 通路） | `.global`（配 `ld.global.nc`） | `LDG.E.CONSTANT` | 只读、缓存友好 | 不是独立物理内存；需 `__ldg` 或 `const __restrict__` 触发 |

**几条要记牢的边界：**

- "空间"是 **作用域 + 生命周期 + 可见性** 的组合，**不是"速度等级"**。shared 与 global 的差别首先是作用域，其次才是代价。
- **shared 是 block 私有**：既不是 warp 私有，也不是 grid 共享；block 结束即失效，内容不跨 block 传递。
  换个方向说：**"在同一个 SM 上"不等于"能互相看见 shared"**。一个 SM 上常住着多个 block
  （这正是 occupancy 的来源），它们各自有**独立的一份** shared memory——shared 的隔离单位是
  **block**，不是 warp 也不是 SM。两个 block 就算同宿一个 SM，也从对方 shared 里什么也看不见：
  想跨 block 传数据只能走 global；想省一道数据传递，要么把它们变成同一个 block 里的更多线程。
- **local 是每线程逻辑私有，但物理上就是 device memory**（与 global 同一块 DRAM），
  **不是**"每线程一块独占的片上内存"。所以 local 流量会和 global 争 L1/L2/DRAM 带宽。
- **内存层级长这样**（从近到远）：寄存器 → SM 片上 SRAM（这块要么当 shared，要么当 L1），
  → **L2（全 device 一个，线程间不区分）** → device DRAM。
  L2 是**缓存共享**：block A 写过的数据若被 block B 读到，多半是经 L2 完成的——
  但**缓存共享 ≠ 线程同步**。缓存只解决"数据在那里"，不解决"读到的是不是最新写入"：
  顺序与可见性（什么时候可以放心读）要靠域 4 讲的 fence / 原子 / 屏障来建立。
- **register ↔ local 的分界由编译器定**：数组一旦用运行时索引通常被迫进 local，但若索引"可证明在界内"，ptxas 会把 local 数组**提升回寄存器**。
- **未经 `__syncthreads()` 的 shared 跨线程读**不是"可能读到旧值"，而是**数据竞争（UB）**（域 4/6）。
- **constant 的"广播"只对 uniform 地址成立**；每 lane 不同地址会退化成串行访问。
  另注意两处局限：constant 走的是**专门的常量缓存通路**（`LDC/ULDC`），不是普通 load 的路径；
  广播优化是否生效还与**工具链版本和架构**相关，同一份 PTX 在不同代际可能落不同的 SASS。
  texture/read-only 同理：它走 `__ldg` / `ld.global.nc` 的**只读缓存通路**，需要编译器
  能**证明**"这块内存在 kernel 里没人写"（`__ldg` 显式说、`const __restrict__` 让编译器推），
  才切换通路——不是你"想要只读就能用只读通道"。
- 向量类型（`float4` 等）与**对齐**影响合并访问；现代 GPU 的合并粒度是 **32B sector**（域 12）。

## 反例

### 反例 1（主）：`float arr[16]` 到底在哪？——源码说了不算

下面两个 kernel 都用同样写法声明 `float arr[16]`，也都用**运行时**索引去访问，实测结果却完全不同：

| kernel | 索引特点 | regs | stack frame | SASS `LDL`/`STL` | 结论 |
|---|---|---|---|---|---|
| `k_local_promoted` | 索引 `& 15`，**可证明在界内** | 25 | 0 B | **0 / 0** | 整个数组被**提升回寄存器** |
| `k_local_loop` | `volatile` + 长循环动态索引 | 33 | 64 B | 32 / 4 | 真的在 **local memory** |

还有个更深的坑：**PTX 里出现 `ld.local`/`st.local`，并不等于 SASS 真的走 local memory。** 我们最早的一版内核，PTX 里有 `__local_depot`、`st.local.v4`，可 ptxas 把它提升回寄存器后，SASS 里**一条 `LDL/STL` 都没有**。

> 所以"数组 = local memory = 慢"这种直觉是错的：
> 1. 源码写了数组，也可能根本不在 local；
> 2. 就算写进 PTX 的 local，SASS 也可能不是；
> 3. local 是否"慢"要看它是否 L1 命中、以及是否和 global 争带宽。
> **结论：空间是编译器/架构的决定，只能看 SASS 才能确定。**

### 反例 2：constant memory 不是"永远快"，只对同地址广播快

`__constant__ cbuf[128]`，每轮一次读：

- **uniform 地址**（同 warp 所有 lane 读同一地址）：SASS 用 `ULDC UR5, c[0x3][UR9]`——走 **uniform 数据通路**，广播。
- **每 lane 分叉地址**：SASS 用 `LDC R8, c[0x3][R13]`——每 lane 一次，**串行化**。

实测（单 warp；2026-10-08 首测 35.16 vs 91.55 ms，2026-10-11 复测 15.2 vs 39.3 ms——绝对值随 GPU 时钟状态浮动，倍数稳定 ≈ **2.6×**。单 warp 已经能看出差距；warp 越多，串行化的惩罚越被放大）。把 constant 当"万能快速只读内存"是错的。

### 反例 3（附）：shared 也不是"无条件比 global 快"

32 lanes、地址 `lane*stride + r`：`stride=1`（banks 0..31 各不相同）1.95 ms，`stride=32`（每 lane 都落在 bank 0）8.74 ms（2026-10-11 复测；2026-10-08 首测 4.10 / 20.05 ms），**≈4.5–4.9×** 的 bank conflict。shared 的价值是低延迟 + block 私有 + 片上带宽，而不是"任何访问都快"。

## 本机验证（命令 + 观察）

编译前置：`call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`，再 `nvcc -arch=sm_89`。
源码：`experiments/EXP-0003-memory-spaces/scripts/mem_spaces.cu`；一键复现：`scripts\run.ps1`；产物：`tmp/outputs/D03/2026-10-08/`。

### 1) PTX：变量落在哪个 state space

```powershell
nvcc -O3 -arch=sm_89 -ptx -o mem_spaces.ptx mem_spaces.cu
```

观察（`ptx-state-spaces.txt`，节选；本机 2026-10-11 复验逐行核对）：

```
DECL  .const .align 4 .b8 cbuf[512];
DECL  .local .align 16 .b8 	__local_depot1[64];
DECL  .local .align 16 .b8 	__local_depot2[64];
DECL  .local .align 16 .b8 	__local_depot3[64];
DECL  .shared .align 4 .b8 _ZZ14k_shared_chasePKfPflE1s[64];
DECL  .shared .align 4 .b8 _ZZ8k_sharedPfilE1s[4096];
DECL  .local .align 16 .b8 	__local_depot10[32];
MEM   ld.const.f32                 x10
MEM   ld.global.nc.f32             x1
MEM   ld.local.f32                 x55
MEM   ld.shared.f32                x10
MEM   st.local.v4.f32              x4
MEM   st.shared.f32                x2
```

→ 每个空间有独立的**声明**和**指令前缀**；global 空间没有变量声明，只以 `ld.global`/`st.global` 出现。

### 2) SASS：每个 kernel 实际用了哪些内存指令

```powershell
nvcc -O3 -lineinfo -arch=sm_89 -Xptxas -v -o mem_spaces.exe mem_spaces.cu
cuobjdump -sass mem_spaces.exe > sass.txt
```

`-Xptxas -v` 报栈帧（`sass-opcode-histogram.txt` 汇总各 kernel 的内存指令数）：

| kernel | regs | stack | LDL | STL | LDS | STS | LDC | ULDC | LDG.E.CONSTANT |
|---|---|---|---|---|---|---|---|---|---|
| `k_local_promoted` | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `k_local_loop` | 33 | **64 B** | **32** | 4 | 0 | 0 | 0 | 0 | 0 |
| `k_local_chase` | 22 | 64 B | 29 | 4 | 0 | 0 | 0 | 0 | 0 |
| `k_shared_chase` | 11 | 0 | 0 | 0 | 29 | 1 | 0 | 0 | 0 |
| `k_shared` | 17 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | 0 |
| `k_const_broadcast` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | **6** | 0 |
| `k_const_divergent` | 17 | 0 | 0 | 0 | 0 | 0 | **5** | 1 | 0 |
| `k_readonly` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1** |

（另注：每个 kernel 还有一条 `ULDC.64` 参数基址加载/若干参数取值指令——它是 uniform 通路的一部分，与「变量落在哪个空间」的判断无关，表中按相关访存指令统计；本机 2026-10-11 复验 LDG/E 内存指令数逐行一致。）

关键指令原文（`cuobjdump -sass`）：

```
ULDC UR5, c[0x3][UR9]        ; constant 广播 -> uniform 通路
LDC  R8,  c[0x3][R13]        ; constant 分叉 -> 每 lane
LDL  R7,  [R7]               ; local load
STS  [R0.X4], R3             ; shared store
LDG.E.CONSTANT R0, [R2.64]   ; __ldg 只读通路
LDG.E          R7, [R2.64+0x80]
```

### 3) shared bank conflict（计时）

```powershell
.\mem_spaces.exe shared
# shared stride= 1 :    1.947 ms  (300000 iters x 32 lanes)
# shared stride=32 :    8.743 ms
# -> ~4.5x    （2026-10-08 首测 4.102 / 20.049 ms -> 4.9x；绝对值随时钟状态浮动）
```

### 4) global 合并访问（计时，32MB 工作集、65536 线程）

```powershell
.\mem_spaces.exe coalesce
# global stride= 1 :    0.505 ms
# global stride=32 :    6.00 ms
# -> ~12x
```

> 计时绝对值随 GPU 时钟/功耗状态浮动：2026-10-08 首测 0.360 / 9.954 ms（27.6×），2026-10-11 复测 0.5 / 6.0 ms（≈12×）；合并与非合并差**一个数量级**这一结论不变。倍数以同一次运行内的两档对比为准。

> 注意：小工作集（≤ L1/L2）会让差距被缓存掩盖，必须把 buffer 放大到远大于 cache 才看得到合并语义的代价。

### 5) constant 广播 vs 分叉

```powershell
.\mem_spaces.exe const
# const broadcast :   15.245 ms
# const divergent :   39.333 ms
# -> 2.58x    （2026-10-08 首测 35.159 / 91.548 ms -> 2.60x；绝对值随时钟状态浮动）
```

### 6) local vs shared 的 load-to-use 延迟

依赖索引 pointer-chase（2×10⁶ hops）：local(`LDL`) 65.3 ms vs shared(`LDS`) 55.4 ms ≈ **1.18×**（2026-10-11 复测；2026-10-08 首测 108.1 / 91.4 ms，**倍数不变**，绝对值随时钟状态浮动）。方向符合"local 在 L1/device memory 层、shared 在片上 SRAM"，但循环里 float→int 等开销稀释了两者差距，只作方向性证据。

> 关于这个 chase：索引映射是 `j → (7j mod 16)`，7 与 16 互素，所以它是 0..15 上的一个**置换**（又因 7²≡1，是个对合），链会很快落到不动点 `0`/`8` 或进入 2-循环。这不影响结论：循环次数由 `reps` 写死、**不会提前退出**，每次迭代仍是一次依赖当前 `j` 的 load——正是要测的 load-to-use 依赖链；local 与 shared 用的是同一套映射，两边可比。

### 7) compute-sanitizer（**本机失败，已记录**）

```powershell
compute-sanitizer --tool memcheck .\mem_spaces.exe oob   # 越界读 8-float local 数组
```

预期（本机**未观测到**，见下）：报 `Invalid __local__ read`（证明 local 是有界、每线程的真实内存区域）。
本机实测：`Error: Target application terminated before first instrumented API call`。连 NVIDIA 自带的 `compute-sanitizer\sanitizer_smoke.exe` 都报同样错误 → **本机 sanitizer 无法注入**；`ncu` 也因 `ERR_NVGPUCTRPERM` 取不到计数器。两者在 Windows 上都需管理员权限（`EnableDebuggerInterface.bat` / `RmProfilingAdminOnly=0`），不在本 worker 权限内。本域结论不依赖它即可成立（见「未决」）。

## 出处

**CUDA C++ Programming Guide（v12.8, docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/）**

- **§2.3 Memory Hierarchy** —— 六个空间的总览、作用域与生命周期。
- **§3.2.2 Device Memory**；**§3.2.4 Shared Memory**；**§3.2.14 Texture and Surface Memory**。
- **§5.3.2 Device Memory Accesses** —— 合并访问 / 对齐要求。
- **§7.2.1–§7.2.3** `__device__` / `__constant__` / `__shared__` 变量空间说明符；**§7.2.6** `__restrict__`。
- **§7.10 Read-Only Data Cache Load Function**（`__ldg`）；**§7.11 / §7.12** cache-hint 载入 / 存储。
- **§7.15 Address Space Predicate Functions**（`__isGlobal/__isShared/__isConstant/__isLocal`）；**§7.16 Address Space Conversion Functions**（`__cvta_*`）。
- **§7.39 Maximum Number of Registers per Thread**。
- **§9.2.2.1.5 Local Memory** —— 明确 local 位于 device memory 并被缓存。

**PTX ISA 8.7（CUDA 12.8, docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/）**

- **§5.1 State Spaces**：§5.1.1 register、§5.1.3 constant、§5.1.4 global、§5.1.5 local、§5.1.7 shared、§5.1.8 texture（deprecated）。
- **§6.4.1 Addresses as Operands**、**§6.4.1.1 Generic Addressing**。
- **§8.3 State spaces**（内存一致性模型视角下的状态空间）。
- **§9.7.9.1 Cache Operators**；**§9.7.9.8 ld**；**§9.7.9.9 ld.global.nc**；**§9.7.9.11 st**。

## 未决

- **compute-sanitizer 本机不可注入**（driver 596.49 与 sanitizer 2025.1.0 组合；自带 smoke 亦失败）。需管理员执行 `compute-sanitizer\EnableDebuggerInterface.bat` 或换匹配版本后，复测 local OOB 的 `memcheck`（预期 `Invalid __local__ read`）。
- **ncu 计数器权限缺失**（`ERR_NVGPUCTRPERM`）。bank conflict（`l1tex__data_bank_conflicts_*`）、L1/TEX 命中需 `RmProfilingAdminOnly=0` 或管理员；本稿 bank conflict 只有 timing 级证据。
- local vs shared 的延迟差距被循环开销稀释（1.18×）；更干净的测法（多条独立链 / 批处理）待补齐。
- texture 只覆盖了 `__ldg` 只读通路；真实 **texture object / sampler** 路径（线性插值、边界模式等）未做。
- register→local **spill 的触发阈值**（每线程寄存器上限、`__launch_bounds__`）归域 2 / 12；这里只指路。
- L1 与 shared 的 **carveout**（`cudaFuncAttributePreferredSharedMemoryCarveout`）未测。
