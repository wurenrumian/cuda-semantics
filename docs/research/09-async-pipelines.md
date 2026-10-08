---
title: "域 09 · 异步与流水语义"
type: research
status: draft
created: 2026-10-08
updated: 2026-10-08
tags: [cuda, semantics, async, streams, cp.async, pipeline, cuda-graphs, unified-memory, sm_89]
related: [docs/plan/cuda-semantics-topic-map.md, experiments/EXP-0009-async-pipelines/README.md]
---

# 域 09 · 异步与流水语义

> 目标机型：sm_89（Ada, RTX 4060 Laptop）· CUDA 12.8（nvcc V12.8.61 / PTX ISA 8.7）。
> 所有观察都来自本机实测，产物见 `tmp/outputs/D09/2026-10-08/`。

## 一句话

「异步与流水」的语义只围绕一件事：**把"提交"和"执行"分开、把"顺序"显式化**。一次
kernel launch、`cudaMemcpyAsync`、`cp.async` 都只是**入队或发出**；运行时除「同一
stream 内先入队者先完成」和「你自己用事件/同步显式建立的依赖」之外，**不承诺任何顺序，
也不承诺任何并行**。绝大多数错误来自把这两条否定读成了肯定。

## 承诺与边界

### 1. 「异步 API」承诺了什么

- **kernel launch 是异步的**：主机调用返回时，kernel 只是入队，未必已开始/结束。
- **异步只针对"执行"，不针对"错误"**：非法启动配置同步报错；**执行期**错误（越界、非法
  访存）会延后到之后某个同步点才冒出来（见域 10）。异步 API 的返回≠成功。
- **边界**：异步 ≠ 并行，也 ≠ 不阻塞主机。`cudaMemcpyAsync` 在 pageable 内存上会在主机侧
  同步（反例 A）。

### 2. Stream 是唯一的顺序单元

- **同一 stream 内**：操作按提交顺序 FIFO 执行，前一个完成前不开始后一个 —— 这是**承诺**。
- **跨 stream**：**没有隐含顺序**，也**不承诺并行**；能否重叠取决于资源占用（是否还有空余
  SM / 功能单元），见编程指南 §3.2.8.5.5 Overlapping Behavior。
- **默认 stream（stream 0）是历史包袱**，两种语义：
  - **legacy 默认 stream**（默认）：与所有**阻塞** stream 之间**隐式同步** —— null-stream
    上的操作要等所有先前 stream 的工作完成，且之后所有阻塞 stream 的操作又要等它。这就是
    「进了不同 stream 就能并行」被打破的地方（反例 B）。
  - **`--default-stream per-thread`**：stream 0 变成**每主机线程独立**的默认 stream（等价
    于用 `cudaStreamPerThread`），不再做全局隐式同步。
- **边界**：stream 给的只是「顺序」，不是「并行」；grid 把 GPU 铺满时，后发 kernel 只能等。

### 3. 事件（event）= 跨 stream 的显式依赖 + 计时

- `cudaEventRecord(e, s)`：在 stream `s` 里打一个点，标记该点之前工作的完成。
- `cudaStreamWaitEvent(s2, e)`：让 `s2` 暂停到事件 `e` 完成为止 —— 这是建立**跨 stream
  happens-before** 的正确工具，比 `cudaDeviceSynchronize` 精确得多。
- `cudaEventSynchronize` / `cudaEventQuery` 管主机侧；`cudaEventElapsedTime` 用两个事件测
  设备侧耗时。
- **边界**：事件只建立**顺序/可见性**，不搬运数据。

### 4. 同步 API 的作用域

- `cudaStreamSynchronize(s)`：主机等到 `s` 中**已提交**的工作完成（不覆盖之后才提交的）。
- `cudaDeviceSynchronize()`：等待**所有 stream、所有主机线程**的先前工作。
- `cudaStreamQuery(s)`：非阻塞查询，可用来轮询。
- **边界**：同步点同时是**异步错误**浮出水面的点。

### 5. `cudaMemcpyAsync` 与 pinned 内存

- 只有 **page-locked（pinned）** 主机内存才真异步：`Async` 立即返回，DMA 在后台跑。
- **pageable** 主机内存：驱动先把它拷进内部 staging（pinned）缓冲，`cudaMemcpyAsync` 在返回
  前基本已把这次 staging 做完。编程指南的原话：「Async memory copies might also be
  synchronous if they involve host memory that is not page-locked.」
- **边界**：`Async` 还依赖当前上下文/设备（§11.13.1）；D2H 到 pageable 同理。

### 6. `cp.async`：把「搬运」异步化（sm_80+，本机支持）

- `__pipeline_memcpy_async` / `cuda::memcpy_async` / PTX `cp.async`：把 global→shared 的
  拷贝**不经寄存器**直接完成（SASS 是 `LDGSTS`），可与计算重叠。
- **提交 / 等待协议**：`cp.async` → `__pipeline_commit()`（= PTX `cp.async.commit_group`）→
  `__pipeline_wait_prior(N)`（= `cp.async.wait_group N`）。`wait_prior(0)` 表示等到所有未完成组。
- **必须在 wait 之后**才能读那块 shared；否则是在与 async proxy 的写入竞争（反例 C）。
- 变体：`cp.async.ca`（可走 L1，支持 4/8/16B） vs `cp.async.cg`（绕过 L1，仅 16B）。
- **边界**：目标只能是 shared，不到寄存器；完成的可见性由 wait + `__syncthreads()` 建立。

### 7. CUDA Graphs：连「提交」也异步化

- capture：`cudaStreamBeginCapture(s, …)` … `cudaStreamEndCapture(s, &g)`。capture 期间的工作
  **只被记录、不执行**（反例 D 实测 counter=0）。
- `cudaGraphInstantiate` 一次，`cudaGraphLaunch` 可多次 replay，每次 replay 重新执行整张图。
- **边界**：capture 只能在**非 legacy 默认 stream** 上（反例 D 的
  `cudaErrorStreamCaptureUnsupported`）；对 memcpy / 动态并行 / 合作启动有额外约束（§3.2.8.7）。
- 语义价值：把「一批带依赖的工作」一次性提交、降低每次启动的主机开销。

### 8. TMA / `cp.async.bulk`（sm_90，本机不支持）

- 本机 sm_89 **明确不支持**：`cp.async.bulk` / `.mbarrier::complete_tx::bytes` ptxas 直接报
  `requires .target sm_90 or higher`（实测）。
- 概念：bulk copy（整块 global↔shared）、mbarrier 完成机制、tensor 变体 + swizzle，把搬运的
  控制权更多下放给硬件。这里只做概念了解，不在本机运行。

### 9. 统一内存 / UVA（概念）

- 本机 `unifiedAddressing=1`、`managedMemory=1`，但 `pageableMemoryAccess=0`
  （不能直接访存 pageable 主机指针）。
- `cudaMallocManaged` 的内存按需迁移/页错误；UVA 让 host/device 指针落在同一编号空间。
- **边界**：迁移时机与可见性由运行时/驱动决定，不是「零拷贝」承诺。本域未实测，归入域 10。

## 反例

### 反例 A（★核心）——「`cudaMemcpyAsync` 一定是异步的」

在 **pageable** 主机内存上，`cudaMemcpyAsync` 会在主机侧阻塞到 staging 拷贝完成；换
**pinned** 才立即返回。同一份 256 MiB H2D：

| 调用 | 返回耗时（实测） |
|---|---|
| `cudaMemcpy`（同步） | 22.67 ms |
| `cudaMemcpyAsync`，**pageable** | **33.19 ms**（把 stage 做完才返回） |
| `cudaMemcpyAsync`，**pinned** | **0.013 ms**（立即返回） |

差了三个数量级。所以「我用 Async 就能和计算重叠」在 pageable 内存下根本不成立 —— 主机被
卡住，拷贝与 kernel 自然重叠不了。（与 topic map「pinned memory 的可见性/迁移语义」对应。）

### 反例 B（★）——「用了别的 stream，默认 stream 的操作就与它无关了」

同一份源码，只改编译开关，跑「先在显式 stream `s` 上放一个长 kernel，再在默认 stream 上放
一个长 kernel」：

| 编译选项 | 一个 kernel | s + 默认 stream | 比值 |
|---|---|---|---|
| `-default-stream legacy`（默认） | ~158.7 ms | 316.6 ms | **1.99（被隐式串起来）** |
| `--default-stream per-thread` | ~158.7 ms | 165.8 ms | **1.04（真并行）** |

legacy 默认 stream 的隐式同步把两个 stream「缝」在了一起。这是最容易在性能分析里踩的坑：
代码里用了多 stream，却因为默认 stream 参与而并不重叠。

### 反例 C（★）——「`cp.async` 发出后 shared 立即可读」

把 `__pipeline_wait_prior(0)` 去掉后再读 shared，是数据竞争（UB）。实测同一程序连跑 5 次：
**首轮 256/256 元素错，之后 4 轮 0/256**。这恰好说明它是 race 而不是「稳定地错」：能不能对
取决于 `cp.async` 的到达时机。SASS 侧的证据更直观 —— 正确版与去掉 wait 版只差一条等待：

```
cpasync_copy (正确)      : LDGSTS.E.BYPASS.128 → LDGDEPBAR → DEPBAR.LE SB0, 0x0 → BAR.SYNC → LDS.128
cpasync_racy (少了 wait) : LDGSTS.E.BYPASS.128 → LDGDEPBAR →            (无 DEPBAR) → BAR.SYNC → LDS.128
```

`DEPBAR.LE SB0, 0x0` 就是 `__pipeline_wait_prior(0)` 在 SASS 上的落点；删掉它，`LDS` 就会
和尚未落地的 `LDGSTS` 抢同一块 shared。

### 反例 D ——「capture 期间 kernel 会执行」

`cudaStreamBeginCapture` … `cudaStreamEndCapture` 期间发起的两个 kernel **一次都没执行**：
capture 结束时计数器仍是 **0**；`cudaGraphLaunch` 一次变 2，再 replay 一次变 4。capture 是
「录像」，不是「播放」。另外，对 legacy 默认 stream 调 `cudaStreamBeginCapture(0, …)` 直接返回
`cudaErrorStreamCaptureUnsupported`（"operation not permitted when stream is capturing"）。

### 反例 E（澄清）——「进了不同 stream 就一定会并行」

本机 4 个占 1 个 SM 的 kernel：同一 stream **529.9 ms**、4 个 stream **118.8 ms**
（**4.46×**）。看起来很「并行」，但这段代码只证明**可能**并行，不证明**承诺**并行：换成把
GPU 铺满的 grid，后发 kernel 只能排队。多 stream 是**给调度器一个可并行的机会**，不是保证。

## 本机验证（命令 + 观察）

### 环境与命令

```
call E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -lineinfo -arch=sm_89 -o async_pipelines.exe async_pipelines.cu
# 同一源码再编两份，只换默认 stream 语义：
... -default-stream legacy       -o ds_legacy.exe     async_pipelines.cu
... --default-stream per-thread  -o ds_perthread.exe  async_pipelines.cu
E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe -sass async_pipelines.exe > async_pipelines.sass
# sm_90 专属指令探针：期望 sm_89 FAIL / sm_90 OK
... -arch=sm_89 -cubin -o tma_sm89.cubin tma_probe.cu
... -arch=sm_90 -cubin -o tma_sm90.cubin tma_probe.cu
E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe --tool racecheck async_pipelines.exe cpasync
```

一键复现：`pwsh -File experiments/EXP-0009-async-pipelines/scripts/run.ps1`。

### 关键观察

- **`props`**（`props.txt`）：24 个 SM；`asyncEngineCount=1`；`canMapHostMemory=1`；
  `unifiedAddressing=1`；`managedMemory=1`；`pageableMemoryAccess=0`；`l2CacheSize=32 MiB`。
  （`concurrentKernels` 字段已废弃、恒为 1，不要拿它判断并发。）
- **`streams`**（`streams.txt`）：1 stream 529.9 ms / 4 streams 118.8 ms → **4.46×**。
- **`ds`**（`default-stream-legacy.txt` / `default-stream-perthread.txt`）：**1.99 vs 1.04**。
- **`memcpy`**（`memcpy.txt`）：pageable **33.19 ms** vs pinned **0.013 ms**。
- **`cpasync` / `cpasync-racy`**（`cpasync.txt` / `cpasync-racy.txt`）：正确版 0/256 错；
  去掉 wait 首轮 **256/256** 错、其后 0/256。
- **SASS**（`sass/cpasync_sass_compare.txt`）：`cp.async` → `LDGSTS.E.BYPASS.128`（直连
  global→shared）；普通写法 → `LDG.E.128.CONSTANT` + `STS.128`（经寄存器）。正确版多出
  `DEPBAR.LE SB0, 0x0`。
- **`graph`**（`graph.txt`）：capture 后 `counter=0`，replay 后 `2`、`4`；null stream
  capture 返 `cudaErrorStreamCaptureUnsupported`。
- **TMA 探针**（`tma-sm89.log`）：`error : Feature 'cp.async.bulk' requires .target sm_90 or
  higher`；`tma-sm90.log` 干净通过（只编译、不运行）。
- **compute-sanitizer**：**本机不可用**。对任何 CUDA 程序（含最小 `hello`）都只输出
  `Error: Target application terminated before first instrumented API call`，退出码 1。见「未决」。

产物：`tmp/outputs/D09/2026-10-08/`（各模式 `.txt`、`sass/`、构建日志、TMA 探针日志）。

## 出处

- CUDA C++ Programming Guide v12.8（archive 12.8.0）：
  - §3.2.8 Asynchronous Concurrent Execution；§3.2.8.1 Concurrent Execution between Host and
    Device；§3.2.8.2 Concurrent Kernel Execution。
  - §3.2.8.5 Streams → §3.2.8.5.2 Default Stream（含 `--default-stream per-thread`）、
    §3.2.8.5.4 Implicit Synchronization、§3.2.8.5.5 Overlapping Behavior。
  - §3.2.8.7 CUDA Graphs；§3.2.8.8 Events。
  - §3.2.6 Page-Locked Host Memory（pinned）。
  - §2.5 Asynchronous SIMT Programming Model（TMA 概念）。
  - §19 Unified Memory Programming；§11.13.1 cudaMemcpyAsync Current Context/Device Sensitivity。
- PTX ISA v8.7（CUDA 12.8）：
  - §9.7.9.25 Asynchronous Copy；§9.7.9.25.1 Completion Mechanisms for Asynchronous Copy
    Operations（§9.7.9.25.1.2 Mbarrier-based mechanism）；§9.7.9.25.2 Async Proxy。
  - §9.7.9.25.3.1 `cp.async`；§9.7.9.25.3.2 `cp.async.commit_group`；
    §9.7.9.25.3.3 `cp.async.wait_group` / `cp.async.wait_all`。
  - §9.7.9.25.4.1 `cp.async.bulk`（sm_90）；§9.7.9.25.5.2 `cp.async.bulk.tensor`（sm_90）。

## 未决

- **compute-sanitizer 在本机不可用**（2025.1.0 + driver 596.49，对最小程序也注入失败）。
  本域因此**没有** racecheck/synccheck 证据，反例 C 靠「错值 + SASS 缺 `DEPBAR`」佐证。需
  在别处补装/换版本后重验，或把「工具在本机失效」写进基线（域 0）与 topic map。
- **`ncu` / `nsys` 未跑**：重叠率、copy engine 利用率、SM 占用等只用了墙钟时间；时间线证据
  待补（域 12 的性能语义会用到）。
- **统一内存未实测**：`cudaMallocManaged` / prefetch / 迁移语义待做（归域 10）。
- **cp.async 只在单 stage 上验证**：多 stage double buffering 的 `wait_prior(N)` 语义、与
  `__syncthreads()` 的组合，以及 `cp.async.ca` vs `cp.async.cg` 的差异尚未展开。
- **Windows/WDDM 的并发上限**：`asyncEngineCount=1` 对 H2D/D2H 重叠的影响、WDDM 批处理对
  stream 并发的限制，只做了定性观察。
- 反例 E 的「铺满 GPU 则不再重叠」没做出对照实验，待补一个 grid 铺满版。
