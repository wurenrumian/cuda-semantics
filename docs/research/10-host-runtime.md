---
title: 域 10 — 宿主侧运行时 / 启动语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-11
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

## 承诺与边界

### 1. 启动配置 `<<<grid, block, smem, stream>>>`

- **承诺**：配置在**提交时同步校验**。非法配置当场返回错误，kernel 根本不会执行。
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
- 本机单卡（RTX 4060 Laptop），多 GPU / peer access 只在概念层涉及。

## 反例

### 反例 1（主，想当然会错）：launch 之后立刻查错，显示 "no error"，可 kernel 马上要崩

错误直觉：`kernel<<<...>>>()` 出错的话，紧接着 `cudaGetLastError()` 就能发现。

实际：`<<<>>>` 是**异步提交**。本机 `deref_null<<<1,1>>>(nullptr)`（向空指针写值）后
立刻 `cudaGetLastError()` 得到 **`no error (0)`**，直到 `cudaDeviceSynchronize()` 才报出
**`an illegal memory access was encountered (700)`**（`async.txt`）。
→ 只在 launch 后查一次错，等于没查执行错误。

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

**10. compute-sanitizer 本机无法 attach** — `memcheck-async.txt`（连 NVIDIA 自带的
`sanitizer_smoke.exe` 也报同一错）：

```text
========= Error: process didn't terminate successfully
========= Error: Target application terminated before first instrumented API call
```

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

- **compute-sanitizer 在本机不可用**：`E:\NVIDIA\CUDA\v12.8\compute-sanitizer\` 下的
  `sanitizer_smoke.exe` 自己都注入失败（"Target application terminated before first
  instrumented API call"）。需要排查（驱动 596.49 / 注入路径 / 安全软件）。恢复后本域的
  非法访存可拿到更强的定位，但仍不改变"异步错误只在同步点出现"的结论。
- **并发度只做了逻辑证明，没测真重叠**：本域用自旋标志位证明了 default stream 的隐式串行，
  但没测"两个 non-blocking stream 是否真在同一时刻占用 SM"。要定量需 `nsys` 时间线。
- **CUDA 12 runtime 模块级 API**（`cudaLibrary` / `cudaKernel` / `cudaLibraryLoadData`）
  未实测；本域只确认 runtime 不再暴露旧的 `CUmodule` 风格句柄。
- **device-side graph launch / programmatic dependent launch (PDL)** 属于域 9 / 域 12 交界，
  本域未展开。
- `cudaFuncSetAttribute` 对 occupancy 的连锁影响（域 2）只点到为止。
