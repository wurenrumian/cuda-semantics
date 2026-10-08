---
title: 域 03「内存空间」语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-08
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

CUDA 有六个内存空间（**register / local / shared / global / constant / texture**），每个变量或访存都落在**恰好一个**空间里；这个空间决定了它的**作用域（谁能看见）、生命周期（活多久）、物理落点与访问代价**——而"落进哪个空间"并不完全由源码写法决定，它常常是**编译器根据访问模式选择、必须在 PTX/SASS 层面核对**的结果。

## 承诺与边界

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
- **local 是每线程私有，但物理上在 device memory**（= global 所在的那块内存），只是 per-thread 且走 L1。所以 local 流量会和 global 争 L1/L2/DRAM 带宽。
- **register ↔ local 的分界由编译器定**：数组一旦用运行时索引通常被迫进 local，但若索引"可证明在界内"，ptxas 会把 local 数组**提升回寄存器**。
- **未经 `__syncthreads()` 的 shared 跨线程读**不是"可能读到旧值"，而是**数据竞争（UB）**（域 4/6）。
- **constant 的"广播"只对 uniform 地址成立**；每 lane 不同地址会退化成串行访问。
- 向量类型（`float4` 等）与**对齐**影响合并访问；现代 GPU 的合并粒度是 **32B sector**（域 12）。

## 反例

### 反例 1（主）：`float arr[16]` 到底在哪？——源码说了不算

两个 kernel 的源码**完全一样**地声明 `float arr[16]` 并用**运行时**索引访问它，实测却完全不同：

| kernel | 索引特点 | regs | stack frame | SASS `LDL`/`STL` | 结论 |
|---|---|---|---|---|---|
| `k_local_promoted` | 索引 `& 15`，**可证明在界内** | 25 | 0 B | **0 / 0** | 整个数组被**提升回寄存器** |
| `k_local_loop` | `volatile` + 长循环动态索引 | 33 | 64 B | 32 / 4 | 真的在 **local memory** |

更深的坑：**在 PTX 里看到 `ld.local`/`st.local`，不等于 SASS 真的走 local memory。** 我们最早的一版内核，PTX 里有 `__local_depot`、`st.local.v4`，但 ptxas 把它提升回寄存器后，SASS 里**一条 `LDL/STL` 都没有**。

> 所以"数组 = local memory = 慢"这种直觉是错的：
> 1. 源码写了数组，也可能根本不在 local；
> 2. 就算写进 PTX 的 local，SASS 也可能不是；
> 3. local 是否"慢"要看它是否 L1 命中、以及是否和 global 争带宽。
> **结论：空间是编译器/架构的决定，只能看 SASS 才能确定。**

### 反例 2：constant memory 不是"永远快"，只对同地址广播快

`__constant__ cbuf[128]`，每轮一次读：

- **uniform 地址**（同 warp 所有 lane 读同一地址）：SASS 用 `ULDC UR5, c[0x3][UR9]`——走 **uniform 数据通路**，广播。
- **每 lane 分叉地址**：SASS 用 `LDC R8, c[0x3][R13]`——每 lane 一次，**串行化**。

实测（单 warp）：broadcast 35.16 ms vs divergent 91.55 ms ≈ **2.6×**。单 warp 已经能看出差距；warp 越多，串行化的惩罚越被放大。把 constant 当"万能快速只读内存"是错的。

### 反例 3（附）：shared 也不是"无条件比 global 快"

32 lanes、地址 `lane*stride + r`：`stride=1`（banks 0..31 各不相同）4.10 ms，`stride=32`（每 lane 都落在 bank 0）20.05 ms，**4.9×** 的 bank conflict。shared 的价值是低延迟 + block 私有 + 片上带宽，而不是"任何访问都快"。

## 本机验证（命令 + 观察）

编译前置：`call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`，再 `nvcc -arch=sm_89`。
源码：`experiments/EXP-0003-memory-spaces/scripts/mem_spaces.cu`；一键复现：`scripts\run.ps1`；产物：`tmp/outputs/D03/2026-10-08/`。

### 1) PTX：变量落在哪个 state space

```powershell
nvcc -O3 -arch=sm_89 -ptx -o mem_spaces.ptx mem_spaces.cu
```

观察（`ptx-state-spaces.txt`）：

```
DECL  .const .align 4 .b8 cbuf[512];
DECL  .local .align 16 .b8 __local_depot0[64];
DECL  .local .align 16 .b8 __local_depot1[64];
DECL  .shared .align 4 .b8 _ZZ8k_sharedPfilE1s[4096];
MEM   ld.const.f32      st.local.f32/v4    ld.local.f32
MEM   ld.shared.f32     ld.global.f32      ld.global.nc.f32   st.global.f32
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
# shared stride= 1 :    4.102 ms  (300000 iters x 32 lanes)
# shared stride=32 :   20.049 ms
# -> 4.9x
```

### 4) global 合并访问（计时，32MB 工作集、65536 线程）

```powershell
.\mem_spaces.exe coalesce
# global stride= 1 :    0.360 ms
# global stride=32 :    9.954 ms
# -> 27.6x
```

> 注意：小工作集（≤ L1/L2）会让差距被缓存掩盖，必须把 buffer 放大到远大于 cache 才看得到合并语义的代价。

### 5) constant 广播 vs 分叉

```powershell
.\mem_spaces.exe const
# const broadcast :   35.159 ms
# const divergent :   91.548 ms
# -> 2.60x
```

### 6) local vs shared 的 load-to-use 延迟

依赖索引 pointer-chase（2×10⁶ hops）：local(`LDL`) 108.1 ms vs shared(`LDS`) 91.4 ms ≈ **1.18×**。方向符合"local 在 L1/device memory 层、shared 在片上 SRAM"，但循环里 float→int 等开销稀释了两者差距，只作方向性证据。

### 7) compute-sanitizer（**本机失败，已记录**）

```powershell
compute-sanitizer --tool memcheck .\mem_spaces.exe oob   # 越界读 8-float local 数组
```

预期：报 `Invalid __local__ read`（证明 local 是有界、每线程的真实内存区域）。
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
