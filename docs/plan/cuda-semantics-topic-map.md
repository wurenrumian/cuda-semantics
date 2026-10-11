---
title: CUDA 语义学习地图
type: plan
status: draft
created: 2026-10-08
updated: 2026-10-11
tags: [cuda, semantics, simt, memory-model, atomics, learning]
related: []
---

# CUDA 语义学习地图

> 目标：把 CUDA 里需要"当承诺来理解"的语义（不是 API 用法）逐域过一遍。
> 每域记：**是什么 / 关键概念 / 本机相关性 / 怎么验证**。

## 0. 本机基线（决定哪些语义适用）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop，**sm_89 (Ada)** |
| 显存 | 8 GB |
| 驱动 | 596.49 |
| CUDA Toolkit | 12.8（nvcc V12.8.61） |
| PyTorch | 2.11.0+cu128（`E:\venvs\pytorch312`），cuda_avail=True |
| 工具 | nvcc / ptxas / cuobjdump / nvdisasm ✓；**compute-sanitizer 2025.1.0 ✓**（在 `E:\NVIDIA\CUDA\v12.8\compute-sanitizer\`）；Nsight Compute 2025.1.0 ✓；Nsight Systems 2024.6.2 ✓ |
| MSVC | VS2022 Community `E:\Microsoft Visual Studio\2022\Community`（nvcc 编译前须先 `vcvars64.bat`） |

**基线带来的三条硬约束**（贯穿全图）：

1. **Volta+（本机 sm_89）是独立线程调度（ITS）** → 依赖"warp 隐式锁步"的旧代码在这里是未定义行为。
2. **sm_89 < sm_90** → **无** TMA、thread block cluster / DSMEM、128-bit 原子、`cp.async.bulk`。这些当概念了解，不在本机跑。
3. **工具链已安装** → `compute-sanitizer`（racecheck / synccheck / initcheck）本来用于验证内存可见性与竞争假设；但本机**非管理员会话下它无法注入目标进程**，`ncu` 计数器也被拒（`ERR_NVGPUCTRPERM`）。这两项待有权限环境复核，只当基线里的"受限项"。

---

## 1. SIMT 执行模型语义

- warp = 32 线程；SIMT 的"锁步"到底承诺了什么。
- **分支发散与重汇聚**：active mask、divergence、reconvergence stack。
- **独立线程调度（ITS，Volta+）**：线程可各自向前推进；`__syncwarp` 成为显式同步。
- warp 级原语：`__shfl_sync` / `__shfl_up|down|xor`、`__ballot_sync`、`__any_sync` / `__all_sync`、`__match_any_sync`、`__reduce_*_sync`（sm_80+）。
- `__activemask()` 的语义与**不可依赖**之处。
- **经典坑**：`if (lane == 0) x = ...; y = __shfl(...)` 在 ITS 下不保证参与线程的屏蔽字。

**验证**：`__syncwarp` vs 无同步的对比 kernel；看 SASS 里的 `BSSY/BSYNC`、`WARPSYNC`。

## 2. 线程层次与调度语义

- thread / block / grid；`threadIdx`、`blockIdx`、`blockDim`、`gridDim`。
- **block 间无执行顺序保证**；不能假设 block 0 先跑完。
- block 内也无固定 warp 顺序；`__syncthreads` 之前的可见性由屏障建立。
- **occupancy 语义**：寄存器 / 共享内存 / 线程上限如何决定并发 warp 数。
- `__launch_bounds__` 与寄存器压力。
- 唯一能跨 block 同步的手段：cooperative groups `grid.sync()`、cluster（sm_90）、或自建原子协议。

**验证**：把 block 完成顺序打印出来（会随运行变化）；用 `ncu` 看 occupancy。

## 3. 内存空间语义

- 六种空间：**global / shared / local / register / constant / texture**。
- 每个空间的**作用域、生命周期、可见性**：shared 是 block 私有、寄存器是线程私有、local 实际落在 global。
- shared memory **bank 与 bank conflict**；padding/重排为何有用。
- 合并访问（coalescing）语义与访存对齐要求（`float4` 等向量类型）。
- constant memory 的广播语义；`__restrict__` 对别名的承诺。

**验证**：`cuobjdump -sass` 看 `LDS/STS` 与 `LDG/STG`；`ncu` 看 L1/TEX 命中与 bank conflict 指标。

## 4. 内存一致性模型（★核心，最难）

- **CUDA 内存模型是弱模型**：需要显式同步才建立 happens-before。
- **作用域（scope）**：`_block` < `_device` < `_system`；原子与 fence 都带作用域。
- **顺序（order）**：`relaxed` / `acquire` / `release` / `acq_rel` / `seq_cst`，与 C++ `<atomic>` 的 memory_order 对应。
- **fence**：`__threadfence_block()` / `__threadfence()` / `__threadfence_system()` 各自保证什么。
- `volatile` 在 CUDA 里**不是**同步原语，只阻止编译器优化。
- 与 C++ 内存模型的关系：libcu++ 把 `<atomic>`/`<barrier>` 语义映射到 CUDA。
- 经典模式：**生产者-消费者**、**标志位 + fence**、**acquire mask 搜索**（CUB 的 decoupled look-back）。

**验证**：`cuda::atomic_ref` 各 order 对比；`compute-sanitizer --tool racecheck`（本机注入受限，待复核）；PTX 里看 `ld.acquire.gpu` / `st.release.gpu` / `fence.sc.gpu`。

## 5. 原子语义

本机 header 实测存在的 device intrinsics：

`atomicAdd` `atomicSub` `atomicExch` `atomicMin` `atomicMax` `atomicInc` `atomicDec` `atomicCAS` `atomicAnd` `atomicOr` `atomicXor`，以及 `*_block` / `*_system` 变体。

- **原子性 vs 内存序是两件事**：默认 intrinsic 是 relaxed 的；要序就配 fence 或用 libcu++。
- `atomicCAS` 是通用原子（可自建任意 RMW）；compare-and-swap 循环的语义。
- **浮点原子**：`atomicAdd(float/double)`；本机 sm_89 支持，但 128-bit（`.b128`）要 sm_90。
- `cuda::atomic` / `cuda::atomic_ref`（libcu++）：可指定 scope + order，是"语义正确"的写法。
- PTX 层：`atom`（返回旧值）与 `red`（不返回值）指令，`.relaxed/.acquire/.release/.acq_rel`，`.cta/.gpu/.sys`。

**验证**：无锁计数器 / 直方图 / 自旋锁；`ncu` 看原子操作吞吐与热点。

## 6. 同步原语语义

- `__syncthreads()`：block 内全屏障，隐含内存序。
- **`__syncwarp(mask)`**：Volta+ 下真正的 warp 屏障；mask 里的线程必须都到达。
- `bar.sync` / **命名屏障**（`barrier.sync id, count`）在 PTX 层。
- **cooperative groups**：`this_thread_block()`、`tiled_partition`、`grid.sync()`（需 cooperative launch）。
- `mbarrier`（Ampere+）：异步屏障，配合 `cp.async` 做流水。
- `cuda::barrier` / `cuda::latch` / `cuda::counting_semaphore`（libcu++）。
- **cluster / distributed shared memory（sm_90）**：本机不支持，概念了解。

**验证**：手写 `bar.sync` 死锁；`__syncwarp` 与隐式锁步的差异；`compute-sanitizer --tool synccheck`（本机注入受限，待复核）。

## 7. 指令级 / PTX 语义

- 弱访存 `ld` / `st` 与 **cache operator**：`.ca` `.cg` `.cs` `.lu` `.cv` `.wt`。
- 强访存：`ld.acquire` / `st.release`、`.relaxed`、`membar`。
- 只读数据路径 `__ldg` 与 `const __restrict__` 的语义。
- **inline PTX** 与操作数约束（学语义的最快显微镜）。
- SASS 层：`LDG/STG/LDS/STS`、`ATOM/RED`、`MEMBAR`、`BSSY/BSYNC/WARPSYNC`。
- 编译器是否会重排、合并、消除访存（配合 4、11 看）。

**验证**：`nvcc -ptx`、`nvcc -cubin` + `cuobjdump -sass`、`nvdisasm -c`。

## 8. 浮点与数值语义

- **FMA 收缩**：`--fmad`（默认开）会改变数值结果；`__fmaf_rn` 是显式版本。
- **fast math**：`-use_fast_math`（= `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true`）改了哪些语义。
- **denormal / FTZ**、舍入模式、`__fadd_rn` 系列 intrinsic。
- NaN / Inf 传播、`isfinite` 等。
- 半精度 `__half` / `__bf16`、fp8、tensor core（wmma / mma）——sm_89 支持 fp8 tensor core。
- 精度与可复现性：`atomicAdd(float)` 的结果依赖于顺序。

**验证**：同 kernel 加不加 fast math 跑出不同 bit；对比 `-G`（禁优化）与 `-O3`。

## 9. 异步与流水语义

- **streams**：默认 stream vs per-thread default stream（`--default-stream`），跨 stream 无隐含顺序。
- **events** 与跨 stream 依赖；`cudaStreamSynchronize` / `cudaDeviceSynchronize` 的语义边界。
- **`cp.async`（sm_80+）** → `cuda::memcpy_async` + `cuda::pipeline`（double buffering）。
- **TMA / `cp.async.bulk`（sm_90）**：本机不支持，概念了解。
- **CUDA Graphs**：capture 与 replay 的执行语义。
- pinned memory、UVA / unified memory 的可见性与迁移语义。

**验证**：`nsys` 看时间线；用 event 测重叠；memory pool 行为。

## 10. 宿主侧运行时 / 启动语义

- 启动配置 `<<<grid, block, smem, stream>>>` 与动态并行。
- 错误传播：异步错误何时被 `cudaGetLastError` / `cudaPeekAtLastError` 捕获。
- 设备、上下文、模块（`cudaModule`）、多 GPU（本机单卡）。
- 内核执行顺序、并发内核的隐式假设。

**验证**：故意制造非法访问看错误在哪一步冒出来。

## 11. 编译器行为与未定义行为（UB）

- 别名规则：`__restrict__`、`const`、指针类型转换。
- `volatile` vs 原子 vs fence —— 三者的边界。
- **数据竞争即 UB**：不是"读到一个旧值"而是整个程序无意义。
- 未初始化读取；越界；对已释放/已退出线程的假设。
- 优化会不会删除"看起来没用"的访存 / 死循环。

**验证**：同上，配合 sanitizer 与 PTX 对照。

## 12. 架构 / 微架构相关的语义

有些语义不是 ISA 抽象层的，而是**绑在具体 GPU 的执行逻辑与架构实现**上。要点：分清"ISA 承诺的" vs "架构决定的"。

- **SM / 分区结构**：一个 SM 分若干 sub-partition（warp scheduler），每个有独立的 warp 槽位、寄存器文件、调度/发射逻辑。→ warp 被**静态驻留**在某个 sub-partition 上。
- **Warp scheduler 与发射语义**：每周期每 scheduler 发射有限指令；靠多 warp 切换**隐藏延迟**（延迟隐藏是架构行为，不是语义承诺）。
- **独立线程调度（ITS, Volta+）是架构实现**：per-thread PC + 重汇聚栈 → 才有了 `__syncwarp` 的必要性（域 1、6）。
- **Warp 大小 32 是 ISA 常量**（跨架构固定）；而 **block/warp 的调度顺序、延迟、吞吐都不是承诺**。
- **内存层级的架构差异**：L1/共享内存统一且可配置划分（carveout）、L2、**32B sector** 决定的合并访问粒度、各代 L1 容量/带宽不同。
- **原子的落点**：global 原子在 L2 的原子单元解析，shared 原子在共享内存解析 → 吞吐与竞争语义随架构变。
- **功能单元**：SFU / DP / tensor core 是架构相关的；mma/wmma 的 **fragment 布局**由架构+cuda core 代数决定（sm_89 支持 fp8 tensor core）。
- **访存路径**：`cp.async` 有两个 cache-hint，`cp.async.cg` 绕过 L1、`cp.async.ca` 不绕（SASS 上是 `LDGSTS.E.BYPASS` vs `LDGSTS.E`；sm_80+）；`__ldg` 只读路径依赖 L1/TEX 结构。
- **目标的代数后缀**：`-arch=sm_89` vs `sm_89a` / `sm_XXf` 族目标 → 决定能否用族特有指令。
- **不可依赖的"实现细节"**：bank 冲突代价、原子吞吐、发散代价、寄存器→占用率曲线——都是**性能语义**，会随架构变，别写成正确性假设。
- **可查询的架构事实**：`cudaDeviceProp`（`multiProcessorCount`、`warpSize`、`maxThreadsPerMultiProcessor`、`regsPerMultiprocessor`、`sharedMemPerBlockOptin`、`l2CacheSize`…）、`%smid`。

**验证**：读 `cudaDeviceProp` 打印架构参数；同一 kernel 在 `-arch=sm_75/80/89` 下对比 SASS 与行为；`ncu` 看 scheduler/内存层级指标。

---

## 建议学习阶段（每阶段都动手）

| 阶段 | 主题 | 产出 |
|---|---|---|
| 1 | SIMT 与发散（域 1、2、3） | 一个 kernel + SASS 观察，理解 active mask / 合并访问 |
| 2 | 同步（域 6） | `__syncthreads` / `__syncwarp` / 命名屏障 的最小实验 |
| 3 | **内存模型与原子（域 4、5）** | 生产者-消费者、标志位+fence、CAS 计数器；PTX 对照 |
| 4 | 浮点语义（域 8） | fast-math / fmad 对比实验 |
| 5 | 异步与流水（域 9） | double buffering / cp.async 实验 |
| 6 | 编译器与 UB（域 11） | 复现优化引发的"语义惊变" |
| 7 | 架构语义（域 12） | 打印 `cudaDeviceProp`，跨 `-arch` 对比 SASS |

## 实验手段（本机清单）

- 编译观察：`nvcc -ptx`、`nvcc -cubin` + `cuobjdump -sass` / `nvdisasm -c`。
- 动手最小 repro：每个语义点一个小 kernel，先跑通再对照。
- profiler：`ncu.exe`（Nsight Compute 2025.1.0）/ `nsys.exe`（Nsight Systems 2024.6.2）。
- `compute-sanitizer`（`E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe`）：`racecheck` / `synccheck` / `initcheck` —— 验证本图 4、6、11 域的直接手段（**本机非管理员会话下注入失败，待有权限环境复核**）。

## 未决

- ~~学习成果的落地形态~~ 已定（2026-10-08）：`docs/research/` 13 篇 + `docs/experiments/` 实验页 + 本机 runner（见 `INTENT.md`）。
- 是否需要读官方文档并做笔记（CUDA C++ Programming Guide 的 programming model + memory model 章）。
