---
title: 线程层次与调度 —— thread / block / grid 的承诺与边界
type: research
status: draft
created: 2026-10-08
updated: 2026-10-11
tags: [cuda, semantics, thread-hierarchy, scheduling, occupancy, launch-bounds, sm_89]
related:
  - docs/plan/cuda-semantics-topic-map.md
  - experiments/EXP-0002-thread-hierarchy/README.md
  - tmp/outputs/D02/2026-10-08/
---

# 02 线程层次与调度

> 目标机：RTX 4060 Laptop / **sm_89 (Ada)**，CUDA **12.8**（nvcc V12.8.61）。
> 配套实验：`experiments/EXP-0002-thread-hierarchy/`，产物 `tmp/outputs/D02/2026-10-08/`。

## 一句话

CUDA 只承诺**静态编号**（`threadIdx ⊂ blockIdx`，每维范围固定）和"**block 被放到某个 SM 上执行**"；它**不承诺任何执行顺序**——block 之间、block 内 warp 之间都是"任意顺序、可并行可串行"，占用率是**资源与架构的函数**而非语言承诺。

## 承诺与边界

| | 承诺（可依赖） | 不承诺（不可依赖） |
|---|---|---|
| 编号 | `threadIdx` / `blockIdx` / `blockDim` / `gridDim` 在一次 launch 内是常量；`threadIdx.x` 变化最快地线性化线程（x→y→z） | 编号 → warp → sub-partition 的调度映射 |
| 执行 | grid 里每个线程都会被执行；一个 block 被调度到某个 SM 上执行 | **block 的启动/完成顺序**（block 0 不一定先跑）；**block 内 warp 的发射顺序**；哪一代 warp 先推进 |
| 资源 | warp 大小 = 32（ISA 常量）；shared memory 为 block 私有 | 一个 SM 上同时驻留多少 block（由寄存器/shared/线程上限决定）；`%smid` 是稳定的物理 SM 编号 |
| 同步 | `__syncthreads()` 建立 **block 内**的 happens-before（域 6） | 任何**跨 block** 的顺序或可见性；唯一被承诺的跨 block 屏障是 cooperative launch 的 `grid.sync()`（域 6）。cluster 是 sm_90，本机没有 |

`__syncthreads()` 还有一个前提：它等价于整块的 `bar.sync 0`，**不带参与者数量**，所以块内所有未退出的线程都必须到达；
只有一部分线程到达是 UB（实际表现是永久等待）。要只同步一部分线程，得用带 `count` 的命名屏障 `bar.sync id, count`，
而 `count` 必须是 warp 大小的整数倍。

边界：**host 端 `<<<...>>>` 返回 ≠ kernel 完成**；`cudaDeviceSynchronize()` 之后才有全序。跨 block 的可见性要靠 device-scope 的原子/fence（域 4、5），不是 block-scope 的 `__syncthreads()`。

## 反例

### 反例 1（核心）：以为 "block 会按 `blockIdx` 顺序启动/完成"

**想当然**：grid 是 `blockIdx = 0,1,2,...`，硬件应该从 0 号块开始按序跑，所以 "block 0 先准备好数据，其它 block 再读" 是安全的。

**事实**：调度顺序**完全没有保证**，而且实测每轮都变。`thread_hierarchy.exe sched`（1024 block）三次运行的启动票顺序（`blockIdx` 序列）：

```
run 0: start==blockIdx? NO | finish==blockIdx? NO | finish==start? NO
  start  order[0..31] : 258 170 51 178 42 242 16 34 6 60 15 33 7 61 ...
run 1: start==blockIdx? NO | finish==blockIdx? NO | finish==start? NO
  start  order[0..31] : 6 276 42 7 241 43 51 8 60 214 25 158 17 145 ...
run 2: start==blockIdx? NO | finish==blockIdx? NO | finish==start? NO
  start  order[0..31] : 159 6 205 42 7 161 43 24 60 33 158 8 192 145 ...
  start-order[0..63] identical to previous run? NO
```

三次都**不是** `0,1,2,...`，彼此也不相同，完成顺序又和启动顺序不同。原因：block 一旦**跑完就腾出槽位**，硬件立刻补一个新 block 进去，于是启动顺序被打乱成"随完成滚动"。

**后果**：任何"block 0 当生产者、其余当消费者"的写法都是 UB（不是"读到旧值"，是整个程序无定义）。要做跨 block 同步只能用 cooperative groups `grid.sync()`，或"标志位 + device-scope 原子/fence"自建协议（域 4、5）。

### 反例 2：以为 "并发 block 数只由块内线程数决定"

**想当然**：我用了 128 线程/块，SM 上限 1536 线程，那就该有 12 个块同时跑。

**事实**：寄存器或 shared memory 往往**先**到顶。本机实测（`occupancy` 模式，块大小都是 128）：

| kernel | regs/thread | shared/block | 结果 |
|---|---|---|---|
| `occ_plain` | 10 | 0 | **12** blocks/SM（1536 线程，线程数到顶） |
| `occ_regs` | 72 | 0 | **7** blocks/SM（896 线程，**寄存器**到顶） |
| `occ_smem` | 12 | 16 KB | **5** blocks/SM（640 线程，**shared**到顶） |

所以"看线程数估并发"是错的：能不能再塞下一个块，看的是寄存器、shared memory、线程数三者里**最先到顶**的那个。

### 反例 3：以为 "`__launch_bounds__(256, 8)` 会保证 8 个块常驻"

**想当然**：第二个参数 `minBlocksPerMultiprocessor` 是硬性保证。

**事实**：它只是**给编译器的 hint**。当请求不可行时，ptxas **静默忽略**。本机 `maxThreadsPerMultiProcessor = 1536`，而 `256 × 8 = 2048 > 1536`，于是：

```
ptxas warning : Value of threads per SM for entry _Z6occ_lbPKfPf is out of range. .minnctapersm will be ignored
```

寄存器并没有被压低（74，与未加 launch bounds 的 `occ_regs` 的 72 基本一样），占用率也**没**变成请求的 8，而是 6 blocks/SM。**对照**：可行请求 `__launch_bounds__(128, 8)`（= 1024 ≤ 1536）确实生效——寄存器被压到 **64**、产生 **48 B spill**，换来 **8** blocks/SM：

```
occ_regs  block=128  regs=72  smem=0      local=0     maxTPB=896  -> 7 blocks/SM (896 threads/SM)
occ_lb    block=128  regs=74  smem=0      local=0     maxTPB=256  -> 6 blocks/SM (768 threads/SM)   # (256,8) 被忽略
occ_lb2   block=128  regs=64  smem=0      local=48    maxTPB=128  -> 8 blocks/SM (1024 threads/SM) # (128,8) 生效
```

即："提高占用率"是靠**用寄存器压力（必要时 spill）换常驻块数**实现的，而且请求本身必须可行。

## 本机验证（命令 + 观察）

```powershell
# 1) 编译环境
call E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
# 源码在仓库内：experiments\EXP-0002-thread-hierarchy\scripts\thread_hierarchy.cu
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -lineinfo -arch=sm_89 -Xptxas -v `
    -o thread_hierarchy.exe experiments\EXP-0002-thread-hierarchy\scripts\thread_hierarchy.cu
# 2) 运行（barrier 的两个演示都是有界的，不会挂住）
.\thread_hierarchy.exe sched
.\thread_hierarchy.exe occupancy
.\thread_hierarchy.exe barrier
# 3) 看 SASS
E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe -sass thread_hierarchy.exe
```

打包脚本：`experiments/EXP-0002-thread-hierarchy/scripts/run.ps1`；产物：`tmp/outputs/D02/2026-10-08/`。

**观察 1 — block 顺序（`sched.txt`）**：见反例 1。`start==blockIdx? NO`、`finish==blockIdx? NO`、跨运行 `identical? NO`。

**观察 2 — 占用率（`occupancy.txt`，本机 `cudaDeviceProp`）**：

```
multiProcessorCount = 24        maxThreadsPerMultiProcessor = 1536
regsPerMultiprocessor = 65536   sharedMemPerMultiprocessor = 102400
sharedMemPerBlock = 49152       maxThreadsPerBlock = 1024      warpSize = 32
```

占用率表见反例 2/3。注意 `occ_smem`：16 KB/块理论可放 6 块，实际 5 块——每块 shared 还有约 1 KB 预留（`102400 / (16384+1024) ≈ 5.88 → 5`），说明占用率不能只按字面容量除。

**观察 3 — SASS（`sass.txt`）**：`threadIdx` / `blockIdx` / `%smid` 都是**特殊寄存器读取**（`S2R`），不是内存访问：

```sass
Function : _Z12sched_kernelPiS_S_Pj
    S2R R0, SR_CTAID.X ;              // blockIdx.x
    S2R R5, SR_TID.X ;                // threadIdx.x
    @P1 BRA 0x180 ;                   // if (threadIdx.x == 0) —— 线程级分支
    ATOMG.E.ADD.STRONG.GPU PT, R11, [R2.64], R11 ;  // atomicAdd
    S2R R13, SR_VIRTUALSMID ;         // %smid -> 虚拟 SM 槽位
    BSYNC B0 ;
```

```sass
Function : _Z8occ_smemPKfPf
    STS [R7], R4 ;                    // shared 写
    BAR.SYNC.DEFER_BLOCKING 0x0 ;     // __syncthreads()
    LDS R0, [R0] ;                    // shared 读
```

命名屏障长这样（`bar.sync 1, 32`）：`@!P0 BAR.SYNC.DEFER_BLOCKING 0x1, 0x20`——障碍号 1、参与者 32 个，
warp 1 因为谓词为假直接跳过，所以不被阻塞。

注意函数头 `EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)`、`%smid` 编译成 `SR_VIRTUALSMID`：sm_89 上拿到的是**虚拟化的 SM 槽位**，用于负载均衡，别把它当稳定物理编号。

**观察 4 — 块内同步（`barrier.txt`，2026-10-11 的新版演示）**：

- `__syncthreads()`（32 线程，0–15 号先做定量的活）：早到的 16–31 号在屏障上等，等的时间就是 0–15 号干活的时间；最后 32 条线程在**同一个时钟点**被放行。复核（2026-10-11，有界重跑）：0 号（晚到）与 16 号（早到）的 elapsed 都是 **403228 cycles**，32 条线程的放行时间差 **0 cycles**。
- 命名屏障 `bar.sync 1, 32`（64 线程的 block，只有 warp 0 参与）：warp 0 到达后被放行，warp 1 不被阻塞。复核：warp 0（lane 0）elapsed **55 cycles**，warp 1（lane 32）elapsed **32 cycles**，warp 1 从未进入屏障。
- 旧版 `barrier` 跑的是"一半线程永不到达 `__syncthreads()`"的死锁 kernel（`while (true) __nanosleep(1000);`），
  那条路径**不会自行结束**，会把 exe、脚本和 runner 一起挂住；本机 sanitizer 又注入失败、没法替它收尾。
  现在 kernel 里不再留无界自旋，`barrier` 换成上面两个定量演示；死锁写法只在 `experiments/EXP-0002-thread-hierarchy/README.md` 里作为反例保留。
  同时 `run.ps1` 给每个子进程加了 host 端看门狗（超时即 kill，并在输出里写 `[WATCHDOG]`），保证脚本一定能返回。

**观察 5 — `compute-sanitizer`（本会话受阻，未取得结果）**：

```powershell
E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe --tool synccheck thread_hierarchy.exe barrier
E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe --tool memcheck  thread_hierarchy.exe sched
```

本 opencode 会话里，对任意 CUDA 二进制（含最小样例）都返回：

```
========= Error: process didn't terminate successfully
========= Error: Target application terminated before first instrumented API call
```

已排除的项：目标直接运行正常（exit 0）；`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface` 已是 `0x1`（文档要求项）；`--target-processes all`、`--require-cuda-init no`、`--launch-timeout 0`、`--injection-path` 均无效；当前会话非管理员。结论：**注入库没进目标进程**，属会话/环境级问题。细节见 `tmp/outputs/D02/2026-10-08/sanitizer-blocked.txt`；留待普通桌面终端复跑。

## 出处

**CUDA C++ Programming Guide（CUDA 12.8）**

- **§1.3 A Scalable Programming Model** —— "each block of threads can be scheduled on any of the available multiprocessors within a GPU, **in any order, concurrently or sequentially**..."
- **§2.2 Thread Hierarchy** —— "Thread blocks are required to execute independently. It must be possible to execute blocks **in any order, in parallel or in series**. This independence requirement allows thread blocks to be scheduled in any order and across any number of cores..."（反例 1 的直接依据）；同一节还给出线程线性化：`threadIdx.x` 变化最快，其次 `y`、`z`。
- **§7.35 Formatted Output** —— "CUDA makes **no guarantees of thread execution order** except at explicit `__syncthreads()` barriers."
- **§4.1 SIMT Architecture** —— warp = 32 线程；**Independent Thread Scheduling**（Volta+：per-thread PC 与调用栈，与域 1 衔接）。ITS 在 CC 7.x 下的细节另见 §16.6.2 Independent Thread Scheduling。
- **§7.38 Launch Bounds** —— `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)`；编译器据此**先定寄存器上限**，超出则降寄存器（可能增加 local memory / 指令数）；超过 `maxThreadsPerBlock` 的 launch 会**失败**；`minBlocksPerMultiprocessor` 不可行时被 ptxas 忽略。
- **§7.6 Synchronization Functions** —— `__syncthreads()`（block 作用域）/ `__syncwarp()`（warp 作用域），保证参与线程之间的内存序。
- **§5.2.3 Multiprocessor Level** —— 占用率 = 常驻 warps / SM 最大值，由寄存器、shared memory、线程上限共同决定（§5.2.3.1 是 Occupancy Calculator 的 API）。

**PTX ISA 8.7（CUDA 12.8）**

- **§10 Special Registers** —— `%tid` / `%ntid` / `%ctaid` / `%nctaid` / `%smid` / `%nsmid` / `%warpid` / `%laneid` 等，只读，反映执行层次状态。
- **§9.7.13.1 `bar` / `barrier`** —— CTA 内屏障 `bar.sync a{, b}`：不给 `b` 就是全 CTA 参与（`__syncthreads()` 的落地）；给 `b` 时**必须是 warp 大小的整数倍**，所以只能按 warp 粒度圈定参与者。

**本机实测**：`tmp/outputs/D02/2026-10-08/`（`build.log` / `sched.txt` / `occupancy.txt` / `barrier.txt` / `sass.txt` / `sanitizer-blocked.txt`）。

## 未决

- **（已处理）旧的 `barrier` 死锁 demo 会挂住**：无界 `while (true) __nanosleep(1000)` 已移除，`barrier` 改成两个有界演示，`run.ps1` 也加了逐进程看门狗（见「观察 4」）。遗留问题：想用 `compute-sanitizer --tool synccheck` 直接抓"发散 `__syncthreads()`"，仍需要在 sanitizer 可用的环境里另写一个可被 sanitizer 收尾的 kernel。
- **compute-sanitizer 未能取数**：需在普通桌面终端（最好管理员）复跑 `synccheck`/`memcheck`；若仍失败，执行 `E:\NVIDIA\CUDA\v12.8\compute-sanitizer\EnableDebuggerInterface.bat`。这是本项目多处验证手段（域 4/6/11）的公共前置，值得优先解决。
- **`%smid` 虚拟化语义**：`SR_VIRTUALSMID` 是否跨 launch 稳定、与 `%nsmid`/物理 SM 的关系，未做实验。
- **计算抢占**（Volta+ computer preemption）：block 被抢占后是否迁移、语义影响，未验证。
- **多维 grid → warp 映射**：文档说 `threadIdx.x` 最快线性化，但"线性序 → warp/sub-partition"的具体映射未在本机观测（与域 12 交叉）。
- **`ncu` 实测占用率**（scheduler/occupancy section）未跑；本稿用的是运行时 API `cudaOccupancyMaxActiveBlocksPerMultiprocessor`。
- **cooperative groups `grid.sync()`**（唯一被承诺的跨 block 同步）留域 6；sm_90 的 cluster/DSMEM 本机不支持，仅概念覆盖。
