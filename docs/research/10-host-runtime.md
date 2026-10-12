---
title: 域 10 — 宿主侧运行时 / 启动语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-12
tags: [cuda, semantics, runtime, launch, streams, error-handling, dynamic-parallelism, sm_89]
related:
  - docs/plan/cuda-semantics-topic-map.md
  - experiments/EXP-0010-host-runtime/README.md
  - tmp/outputs/D10/2026-10-08/
---

# 域 10 — 宿主侧运行时 / 启动语义

> 目标：把「启动一个 kernel」当成一条**语义承诺**来理解 —— 它承诺了什么、
> 不承诺什么、错误与顺序的边界在哪。面向 sm_89 (Ada) / CUDA 12.8。
> 本机证据：`experiments/EXP-0010-host-runtime/`，产物 `tmp/outputs/D10/2026-10-08/`。

## 一句话

`kernel<<<grid, block, smem, stream>>>(...)` **不是一次调用，而是一次异步提交**：
它把工作塞进某个 stream 的队列就立刻返回，宿主侧只同步保证「这条启动描述本身合法」；
kernel 能否正确执行、何时执行、和谁并发，都要到之后某个**同步点**才以错误码或时间线的方式显形。

## 一次 launch 的一生：提交与观察点

把 `<<<>>>` 想成**寄包裹**（提交）而不是**跑腿**（执行），每一步产出的东西不同：

```
 host ─ 配置校验 ─ 参数打包 ─ 入队(stream) ─ 返回
                    │                            │(①同步观察点: <<<>>> 立刻可见)
                    ▼
 stream 队列 ─ 等(前序任务/资源) ─▶ SM 开跑 ─▶ 全部结束
                     (异步错误在这里产生)          │(②异步观察点: sync/查询处浮现)
```

| 步骤 | 发生什么 | 边界 |
|---|---|---|
| **配置校验** | grid/block 尺寸、shared 超额、参数不合法当场回错（观察 1：`invalid configuration argument`） | 只验"配置表"，不预演 kernel 内容 |
| **参数传递** | 参数打进常量 bank（SASS 里 `MOV c[0x0][0x160]` 的来源） | 指针按**值**拷——host 缓冲的后续改动与 device 已拿到的那份无关 |
| **异步提交** | 工作挂进指定 stream 尾部，**立即返回** | ① 只能报错误码 0/失败，不能证明 kernel 跑对 |
| **执行** | 依赖调度/资源；错误（非法访存等）在执行中产生 | 与 host 完全脱钩；互不约束 |
| **完成/观察** | 同步点（`cudaDeviceSynchronize`/stream sync/查询/get）读到 ④ | ② 只有守规矩的调用会**浮出**错误；错误**粘性**（反例 3） |

## 承诺与边界

### 1. 启动配置 `<<<grid, block, smem, stream>>>`

- **承诺**：配置在**提交时同步校验**。非法配置当场返回错误，kernel 根本不会执行。
- 四个实参各管一件实事（直觉例子配 256 线程/1 SM 记忆点）：
  - `grid`（几支队伍）、`block`（每队几人）：总线程 = `grid.x*y*z × block.x*y*z`——写大算例时
    先在心里乘一遍（如可导 256×256 tile 要 `grid(32,32)`、`block(8,8)`）。
  - `smem`：动态 shared 字节数——写比 `cudaFuncSetAttribute` 设的上限还大的数，提交当场回错。
  - `stream`：挂进哪条队列——**决定它与谁有先后关系**（下一节），传 0/省略 = 默认 stream
    （legacy 隐式同步，见反例 2）。
- **边界（本机 `cudaDeviceProp`，见 `props.txt`）**：

  | 项 | 本机值 | 语义边界 |
  |---|---|---|
  | `maxBlockDim` | `[1024, 1024, 64]` | block 总线程数 ≤ 1024（ISA/硬件常量） |
  | `maxGridSize` | `[2147483647, 65535, 65535]` | grid.x 上限 2³¹−1，y/z 上限 65535 |
  | `sharedMemPerBlock` | 49152 (48 KiB) | 动态 smem **默认**上限 |
  | `sharedMemPerBlockOptin` | 101376 (99 KiB) | 调 `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, …)` 后才能超过 48 KiB |
  | `regsPerBlock` | 65536 | 寄存器配额 |
  | `multiProcessorCount` | 24 | 真正决定并发块数的上限之一 |
  | `concurrentKernels` | 1 | 设备支持并发 kernel（旧字段） |

- **`__launch_bounds__(maxThreads, minBlocks)` 是设备侧约束，不是宿主校验**：它编译进 PTX 的
  `.entry` 头（`.maxntid` / `.minnctapersm`），影响寄存器分配与占用率；它**不**改变你在宿主
  `<<<>>>` 里能写多大的 block。见「本机验证」PTX 观察。
- **动态并行（device 侧 `<<<>>>`）的边界**：必须用可重定位设备码编译，即 `-rdc=true`
  并链接设备运行时 `-lcudadevrt`；否则连编译都过不去。且 CUDA 12 起默认走 **CDP2**，
  device 侧 `cudaDeviceSynchronize()` 已不在 `-rdc` 模式下声明（sm_90+ 上更会 fail to load）。

### 2. 错误传播：两条**不同步**的通道

- **同步通道**：启动配置非法、显存不足、参数非法 → 宿主调用当场返回错误。
- **异步通道**：执行期错误（非法访存、非法指令…）。宿主 `<<<>>>` 返回 `cudaSuccess`，
  错误只在之后某个**检查点**（`cudaDeviceSynchronize` / `cudaStreamSynchronize` / 任意返回
  `cudaError_t` 的调用 / `cudaGetLastError`）冒出来，且是**粘性、上下文级**的。
- `cudaGetLastError()`：返回并**清除** last-error 槽位；
  `cudaPeekAtLastError()`：返回但**不清除**。两者都只反映"上下文已记录"的错误，
  **不主动探测**设备；"launch 后 `cudaSuccess`"只说明 launch 合法，不说明 kernel 跑对了。
- 观察点 G 各看什么/清什么（配合生命周期图的 ①②）：

| 调用 | 阻塞 host？ | 能看到什么错误 | 会清除 last-error？ |
|---|---|---|---|
| `cudaGetLastError()` | 否 | 槽里已记录的错误（同步或已浮现的异步） | **是** |
| `cudaPeekAtLastError()` | 否 | 同上 | 否 |
| `cudaStreamQuery(s)` | 否 | s 上已浮现的异步错误 | — |
| `cudaStreamSynchronize(s)` | **是**（至 s 已提交工作完成） | 覆盖面 = stream s | — |
| `cudaEventSynchronize(e)` | **是**（至 event 完成） | 覆盖面 = event 之前的工作 | — |
| `cudaDeviceSynchronize()` | **是**（所有 stream 全部先前工作） | 全设备 | — |

- "谁先看到"没有先后承诺：错误**在上下文记下的那一刻**就位；不同观察点只是不同的"读法"，
  顺带的是否阻塞、覆盖面不同（依赖型调用还兼做依赖边——event 那一行见 09 的依赖图）。
- **边界**：清掉 last-error 槽位 ≠ 修复上下文。本机实测：非法访存后
  `cudaGetLastError()` 取到 700，再 `cudaDeviceSynchronize()` **仍**返回 700 —— 上下文已废。

### 3. stream 与执行顺序

- **同一 stream 内是 in-order**：执行顺序 = 入队顺序（`in-order streams`），不会跳队。
- **不同 stream 之间无顺序承诺**：能并发是**机会性**的，取决于资源与实现；不能把并发
  当作正确性前提。
- **legacy default stream（stream 0 / NULL stream）是 blocking stream**：它会与**所有其它
  blocking stream 隐式互斥同步**。你以为两个 stream 在并行，default stream 一插进来就全串行。
  用 `cudaStreamNonBlocking` 创建非阻塞 stream，或 `--default-stream per-thread` /
  `CUDA_API_PER_THREAD_DEFAULT_STREAM` 改成每 host 线程独立 default stream，才能取消该行为。
- **优先级只是 hint**：不抢占已在执行的工作，也不保证顺序。

### 4. 设备 / 上下文 / 模块

- runtime 为每个 device 维护一个 **primary context**；`cudaSetDevice` 切换当前 device。
- runtime 侧**没有公开的 module/function handle**（`CUmodule` / `CUfunction` 是 Driver API）；
  每个 device 的 default stream 各自独立，跨 device 命令之间无隐式顺序。
- **指针是"带着设备号的东西"**：同一地址值在 device A 有效不代表在 B 有效；跨 device 传指针
  需 UVA（本机 unifiedAddressing=1）支持，且**可见性/访问路径**要与 peer access 设置对上
  （未配 peer 就跨 device 解引用,属提交级/执行级错误两可,看驱动怎么拒）。多 GPU 的顺序/可见性
  靠疯狂显式（event/`cudaDeviceEnablePeerAccess`/`cudaStreamWaitEvent`），别拿单卡直觉平移。
- 本机单卡（RTX 4060 Laptop），多 GPU / peer access 只在概念层涉及。

## 反例

### 反例 1（主，想当然会错）：launch 之后立刻查错，显示 "no error"，可 kernel 马上要崩

错误直觉：`kernel<<<...>>>()` 出错的话，紧接着 `cudaGetLastError()` 就能发现。

实际：`<<<>>>` 是**异步提交**。本机 `deref_null<<<1,1>>>(nullptr)`（向空指针写值）后
立刻 `cudaGetLastError()` 得到 **`no error (0)`**，直到 `cudaDeviceSynchronize()` 才报出
**`an illegal memory access was encountered (700)`**（`async.txt`）。
→ 只在 launch 后查一次错，等于没查执行错误。

**最小正确检查模板**（两处各司其职，都别省）：

```cpp
kernel<<<grid, block, smem, stream>>>(args...);

// ①launch 后即刻查一次：抓"提交级"错误（配置/参数/资源）
cudaError_t e = cudaGetLastError();          // 清槽后立即处理,别让上一错误串到本次
if (e != cudaSuccess) { /* 读 cudaErrorString(e)、规约退出路径,不要吞 */ }

// ②最终同步处再查一次：抓"执行级"错误（非法访存/断言/中止）
cudaError_t s = (sync_mode ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream));
if (s != cudaSuccess) { /* 700 级错误上下文已废:走"重建或退出"分支,不可能原地恢复 */ }
```

两个分支上报错**不是同一类问题**：①修参数/修配置就好；②一经出现、上下文报废（反例 3），
模板里的②必须是"处理并放弃上下文"的路径，写成"清理错误继续跑"就是自欺。

### 反例 2：以为「kernel 之间天然串行」，或以为「开了两个 stream 就并行」

实际有两层反直觉（`streams.txt`）：

- 两个 stream 之间**没有**顺序承诺：并排提交不保证按提交顺序跑。
- 但 legacy default stream 会**强制串行**：default stream 上先放一个自旋 kernel，再用
  **blocking** stream 去写它等的标志位 → 标志位只能在自旋超时后才被写，自旋看到 `flag=0`
  （被隐式串行化）；换成 **non-blocking** stream 才看到 `flag=1`。自旋本身带 `clock64()`
  时间上限（约 0.1 s），所以被串行化那一路是**超时退出**，不会一直空转。

```text
blocking stream    : spin observed flag = 0   (被隐式同步串行化)
non-blocking stream: spin observed flag = 1   (真正并发了)
```

### 反例 3：以为「`cudaGetLastError()` 清掉错误就恢复了」

实际：执行期非法访存是**上下文级粘性错误**。取走 last-error 后，下一次同步**依旧**报 700
（`async.txt` 的 `sync #3`）。这跟"启动配置错误"完全不同 —— 后者清一次就干净了
（`peekget.txt`：`get` 返回 9，`get #2` 立刻回到 `no error`）。

## 本机验证（命令 + 观察）

环境（每篇统一）：

```powershell
call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -lineinfo -arch=sm_89 -o runtime_demo.exe runtime_demo.cu
```

一键复现：`pwsh -File experiments/EXP-0010-host-runtime/scripts/run.ps1`
（另含 `smem_optin.cu`、`dp_parent.cu`、`launchbounds.cu`）。产物目录
`tmp/outputs/D10/2026-10-08/`。以下均为**实跑观察**，非抄文档。

**1. 启动配置是同步校验** — `config.txt`

```text
grid x = 0            : invalid configuration argument (9)
block = 2000 (>1024)  : invalid configuration argument (9)
dynamic smem = 200000 : invalid argument (1)
```

**2. 动态 smem 的 48 KiB 门槛与 opt-in** — `smem_optin.txt`

```text
sharedMemPerBlock=49152  optin=101376
64 KiB BEFORE opt-in : invalid argument
cudaFuncSetAttribute : no error
64 KiB AFTER  opt-in : no error
```

**3. get 清 / peek 不清** — `peekget.txt`

```text
peek #1 : invalid configuration argument (9)
peek #2 : invalid configuration argument (9)   <- 未被清除
get     : invalid configuration argument (9)   <- 返回并清除
get #2  : no error (0)
```

**4. 执行错误的异步 + 粘性** — `async.txt`

```text
right after <<<>>>       : no error (0)                 <- 反例 1
cudaDeviceSynchronize #1 : an illegal memory access ... (700)
cudaDeviceSynchronize #2 : ... (700)
cudaGetLastError         : ... (700)
cudaDeviceSynchronize #3 : ... (700)                    <- 反例 3：清不掉
```

**5. default stream 隐式同步** — `streams.txt`（见反例 2 的 0 / 1 对比）。

**6. 架构边界** — `props.txt`：`maxThreadsPerBlock=1024`、`sharedMemPerBlockOptin=101376`、
`multiProcessorCount=24`、`concurrentKernels=1`。

**7. SASS：执行错误究竟发生在哪** — `cuobjdump -sass runtime_demo.exe`（`cuobjdump_sass.txt`）

```text
Function : _Z10deref_nullPi
  MOV  R2, c[0x0][0x160]     ; kernel 参数从常量 bank0 读出（宿主 launch 传进来的）
  MOV  R3, c[0x0][0x164]
  STG.E [R2.64], R5          ; 123 写入 *p —— 这就是异步 fault 现场
  EXIT
```

值得注意的是自旋 kernel 的 `volatile int*` 被编译成**系统级强访存**（循环由 `clock64()` 计时收口）：

```text
Function : _Z9spin_waitPViPix
  CS2R R4, SR_CLOCKLO            ; start = clock64() —— 自旋的时间上限
  LDG.E.STRONG.SYS R0, [R2.64]   ; volatile 全局读 -> strong, .sys scope
  ...
  @!P0 BRA P1, 0x50              ; 未超时 且 *flag==0 -> 回跳继续自旋
```

（`-res-usage`：`deref_null` REG:8，`nop` REG:4；参数/栈用量见 `cuobjdump_resusage.txt`。）

**8. PTX：宿主 launch 约束如何落到设备模块** — `nvcc -arch=sm_89 -ptx launchbounds.cu`

```text
.visible .entry _Z7boundedPi( .param .u64 _Z7boundedPi_param_0 )
  .maxntid 256, 1, 1
  .minnctapersm 2
.visible .entry _Z9unboundedPi( .param .u64 _Z9unboundedPi_param_0 )
  ; 无 .maxntid / .minnctapersm
```

**9. 动态并行边界** — `dp_build_nordc.log` / `dp_build_rdc.log` / `dp_run.txt`

```text
(无 -rdc) error: kernel launch from __device__ or __global__ functions
                requires separate compilation mode
(有 -rdc -lcudadevrt) 编译链接通过
dp_run: parent sync: no error / child wrote 42 (expect 42)
```

**10. compute-sanitizer（2026-10-12 更新：本域拿到最贴合的判词）** — 解锁（停 Windhawk）后：
`memcheck runtime_demo.exe async` 直接把 700 级错误**指到代码行**：

```text
========= Invalid __global__ write of size 4 bytes
=========     at deref_null(int *)+0x50 in runtime_demo.cu:20
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x0 is out of bounds
```

这正是"非法访存=无效写"的**工具级定位**：行为数据只给错误码 700，memcheck 给出**地址 0x0、
行 20、线程归属**。错误异步性/粘性的结论（反例 1/3）不因此改变——memcheck 只是让"错误发生"的
主张从 API 错误码升格成了指令位置。
（解锁前的失败记录：连自带 `sanitizer_smoke.exe` 都报同一注入错；根因 Windhawk，见域 0。）

## 出处

> 章节号按 **本机 CUDA 12.8** 的《CUDA C++ Programming Guide》与《PTX ISA 8.7》给出。
> 编程指南在 13.x 期间改过编号（异步执行从 §3.2.8 挪到 §2.5），语义未变，但引用以 12.8 为准。

- **CUDA C++ Programming Guide**（12.8）
  - §3.2.8 *Asynchronous Concurrent Execution*
    - §3.2.8.1 *Concurrent Execution between Host and Device* —— 提交即返回，可能早于开始执行
    - §3.2.8.5 *Streams* —— `<<<grid, block, shared_mem_size, stream>>>`；in-order streams
    - §3.2.8.5.2 *Default Stream* —— *Legacy Default Stream* 与 `--default-stream per-thread`
    - §3.2.8.5.3 *Explicit Synchronization* / §3.2.8.5.4 *Implicit Synchronization*
  - §3.2.12 *Error Checking* —— 每 host 线程一份 error state；get 清 / peek 不清；
    `CUDA_LAUNCH_BLOCKING=1`（另见 §18 *CUDA Environment Variables*）；launch 后 `cudaSuccess`
    只证明 launch 合法
  - §3.2.9 *Multi-Device System* —— 每 device 各有自己的 default stream
  - §9 *CUDA Dynamic Parallelism*（含 §9.2.1.5 *Ordering and Concurrency*、
    §9.3.3.2 *Compiling and Linking*、§9.5 *CDP2 vs CDP1*）—— `-rdc=true -lcudadevrt`
- **CUDA Compiler Driver NVCC**（12.8）—— `-rdc` / `-lcudadevrt` 与设备码分离编译。
- **PTX ISA 8.7**（本机 CUDA 12.8 自带）
  - §11.2 *Specifying Kernel Entry Points and Functions*（`.entry` / `.func`）
  - §11.4 *Performance-Tuning Directives*：§11.4.1 `.maxnreg`；§11.4.2 `.maxntid`；
    §11.4.3 `.reqntid`；§11.4.4 `.minnctapersm`
  - §5.1.6 *Parameter State Space*；§5.1.7 *Shared State Space*
- **CUDA Runtime API Reference**：`cudaGetLastError` / `cudaPeekAtLastError` /
  `cudaFuncSetAttribute` / `cudaLaunchKernel` / `cudaStreamCreateWithFlags` / `cudaDeviceSynchronize`。

## 未决

- **（已解决，2026-10-12）compute-sanitizer**：Windhawk 注入冲突解除（见域 0）；本域已拿到
  `deref_null` 的 memcheck 指令级判词（观察 10）。**遗留**：`memcheck` 报过的程序在 sanitizer
  下的粘性表现（如错误后 kernel 继续跑/提前止步）未单独记录——反例 3 的粘性结论来自
  无插桩运行，两者不冲突但未交叉验证。
- **并发度只做了逻辑证明，没测真重叠**：本域用自旋标志位证明了 default stream 的隐式串行，
  但没测"两个 non-blocking stream 是否真在同一时刻占用 SM"。要定量需 `nsys` 时间线
  （`ncu` 需计数器权限 `ERR_NVGPUCTRPERM`，与 Windhawk 无关，见域 0）。
- **CUDA 12 runtime 模块级 API**（`cudaLibrary` / `cudaKernel` / `cudaLibraryLoadData`）
  未实测；本域只确认 runtime 不再暴露旧的 `CUmodule` 风格句柄。
- **device-side graph launch / programmatic dependent launch (PDL)** 属于域 9 / 域 12 交界，
  本域未展开。
- `cudaFuncSetAttribute` 对 occupancy 的连锁影响（域 2）只点到为止。
