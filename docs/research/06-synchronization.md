---
title: 域 06 —— 同步原语（__syncthreads / __syncwarp / 命名屏障）
type: research
status: draft
created: 2026-10-08
updated: 2026-10-11
tags: [cuda, semantics, synchronization, barrier, syncwarp, named-barrier, mbarrier, sm_89]
related:
  - docs/plan/cuda-semantics-topic-map.md
  - experiments/EXP-0006-synchronization/README.md
  - tmp/outputs/D06/2026-10-08/
---

# 域 06 —— 同步原语

> 环境：sm_89（Ada, RTX 4060 Laptop）/ CUDA 12.8（nvcc V12.8.61）/ driver 596.49。
> 本机实验：`EXP-0006`（`experiments/EXP-0006-synchronization/`），产物 `tmp/outputs/D06/2026-10-08/`。

## 一句话

同步原语本质上是**一个「到达 + 可见性」的约定**：屏障只对它名下、且**尚未退出（non-exited）**的线程负责——等这些线程都到达之后，它们在屏障前对 shared/global 的访问才彼此可见。它**不承诺**谁会到达，也**不跨 block**。从 Volta 起（含 sm_89）线程是逐线程执行的，所以"尚未退出的线程（non-exited threads）"这个概念是本域的关键。

## 承诺与边界

### 承诺了什么

| 原语 | 编译成 | 作用域 | 等待谁 | 可见性 |
|---|---|---|---|---|
| `__syncthreads()` | `BAR.SYNC 0`（= `bar.sync 0`，无 count） | block（CTA） | 本 CTA **所有未退出的线程** | 屏障前本线程对 **shared + global** 的访问，对**本 block 所有线程**可见 |
| `__syncwarp(mask)` | `WARPSYNC <mask>` | warp | `mask` 中**未退出的 lane** | 参与线程之间的内存序（可写 shared → syncwarp → 读） |
| `__syncthreads_count/and/or(p)` | `BAR.RED.POPC/AND/OR`（屏障 + 归约融合，SASS 实测） | block | 同 `__syncthreads` | 同 `__syncthreads`，另返回全块归约值 |
| `bar.sync id, count`（内联 PTX，命名屏障） | `BAR.SYNC id, count` | block，`id∈0..15` | 该屏障上到达 `count` 个线程（count 为 warp 倍数） | 同 CTA 范围 |
| cooperative groups `tiled_partition` | 内部用命名屏障（>32 时） | tile | tile 内 | tile 内 |
| `grid.sync()`（`this_grid()`） | grid 屏障 | **整 grid** | 全部 block（需 cooperative launch） | grid 内 |
| `mbarrier`（sm_80+，PTX） | `mbarrier.*` | CTA（shared mem 对象） | phase/pending count + tx-count | 配合 `cp.async` 做异步流水 |
| `cuda::barrier` / `cuda::latch` / `cuda::counting_semaphore`（libcu++） | 映射到上述 | 可按 scope | 语义同 `<barrier>`/`<latch>` | 显式 scope/order |

要点：

- **`__syncthreads()` = block 屏障 + block 范围内存序**。它让 shared 的"写后读"跨整个 block 成立（这是它最常用的用途）。
- **`__syncwarp()` 是真正的 warp 屏障**（ITS/Volta+ 下必须显式写），且**命名了内存序**——`shared[lane]=v; __syncwarp(); r=shared[lane^1];` 是合法的。
- **命名屏障 0..15**：`bar.sync 0`（无 count）就是 `__syncthreads()`；`bar.sync id, count` 让一个 block 里的不同子组各自同步（本机实验用它把 256 线程分成两半）。
- **跨 block 没有隐式同步**：唯一手段是 `grid.sync()`（cooperative launch）、cluster（sm_90，本机无）或自建原子协议 + 带 device scope 的 fence/原子。

### 不承诺什么（边界）

1. **不承诺谁会到达**。你没让线程来，它就不来 —— 后果见下节。
2. **不承诺跨 block 的可见性**。`__syncthreads()` 之后，本 block 的 global 写对其他 block 仍可能不可见；要跨 block 需 `__threadfence()`（device scope）或带 `.gpu` scope 的原子/acquire-release。
3. **不承诺"没挂就是对的"**。发散调用是 UB，是否可见地挂掉取决于时序（见反例）。
4. **命名屏障 `count` 必须是 warp size 的倍数**，且必须恰好等于**会到达的线程数**；`id` 只有 0..15，用光会影响 occupancy（`tiled_partition` >32 时占命名屏障）。
5. **`__syncwarp(mask)` 的 mask 只能命名"已收敛/保证会到"的 lane**；每个调用线程自己那一位必须置位。默认 `0xffffffff` 在发散里不安全。
6. **屏障不产生跨 scope 的 release/acquire 语义**；需要跨观察者时配 fence/原子（域 4）。

## 反例（想当然会错）

**想当然**：「`__syncthreads()` 放进 `if` 分支、只有一半线程到 → 一定死锁」。真相反过来更有教学价值：

> 发散调用 `__syncthreads()` / `__syncwarp()` 是 **UB**，而它**在测试里过不过，取决于没到达的线程是否已经退出**。

两段代码只差 else 分支：

```cpp
// A：没到的那一半 EXIT
__global__ void k_exit(int *out) {
  int t = threadIdx.x;
  if (t < blockDim.x / 2) { out[t] = 1; __syncthreads(); }   // 只等活着的线程
  else                    { out[t] = 2; }                     // 提前 EXIT
}

// B：没到的那一半自旋（活着）
__global__ void k_alive(int *out, volatile int *spin) {
  int t = threadIdx.x;
  if (t < blockDim.x / 2) { out[t] = 1; __syncthreads(); }   // 永远等不到另一半
  else                    { while (*spin == 0) {} out[t] = 2; }
}
```

变体 B 里的自旋是**故意**写的：`*spin` 永远是 0，那半个 block 就永远卡在 `while` 里、永远到不了屏障——这是一个**故意构造、永不终止**的演示，专门用来展示"发散屏障会卡死"，不是代码缺陷。变体 A 则相反，没到屏障的线程直接 `EXIT`。

本机实测（256 线程，sm_89；`-alive` 由 exe 内置看门狗兜底——轮询 50 次 × 10 ms，**截止 ~0.5 s**，实测打印 0.76–0.78 s 判定 wedge，随后 `_exit(0)`，全程不挂 GPU、不挂 shell）：

| 变体 | 结果 |
|---|---|
| A（EXIT） | **FINISHED** —— 屏障照样满足，看起来"没问题" |
| B（自旋） | **WEDGED** —— 看门狗轮询期内 kernel 一直未完成（实测 0.76–0.78 s 判定），块被卡死 |

`__syncwarp()` 同构：`if (lane<16) __syncwarp();` 且另一半 EXIT → **FINISHED**；另一半自旋 → **WEDGED**。

根因在 PTX / 编程指南的措辞里：屏障等的是 **all non-exited threads**（所有尚未退出的线程）。已退出的线程会被从计数里去掉，所以 A 不挂；**但"条件是否在整块 / 整 warp 上一致"仍是 UB 的前提**——A 只是碰巧没暴露而已。真正的危害在于：**同一段错误代码，`else` 只做一点点工作时就"测过了"，一旦换成真正耗时的 `else`（或到了下一版），就会挂死。**

SASS 对照（`cuobjdump -sass`）：

```text
Function : _Z28k_divergent_syncthreads_exitPi      ; 变体 A
        @P0 EXIT ;                                  ; 另一半提前退出
        WARPSYNC 0xffffffff ;                       ; 编译器先强制 warp 收敛
        BAR.SYNC.DEFER_BLOCKING 0x0 ;               ; 只等"未退出"的线程 → 满足

Function : _Z29k_divergent_syncthreads_alivePiPVi  ; 变体 B
 0x0b0: LDG.E.STRONG.SYS R0, [R4.64] ;             ; volatile 自旋（读全局 flag）
        YIELD ;
        @!P0 BRA 0xb0 ;                             ; 另一半永远留在这里
        ...
        BAR.SYNC.DEFER_BLOCKING 0x0 ;               ; 这一半永远等不到 → WEDGE
```

> 附带观察：`volatile` 自旋编成 `LDG.E.STRONG.SYS` —— 它是"强 + system scope"的读，**不等于** acquire，也不构成同步（与域 4/7 呼应）。

## 本机验证（命令 + 观察）

构建/运行（先 `vcvars64.bat`，再 `nvcc -arch=sm_89`）：

```powershell
# 编译 + SASS + 各模式。带 -alive 的自旋模式是故意的永不终止演示：
# exe 内置 host 看门狗（~0.5 s 截止，打印实测秒数）判定 wedge 后退出，
# run.ps1 再给每次运行套 30s 硬超时，所以脚本不会永久挂住。
pwsh -File experiments/EXP-0006-synchronization/scripts/run.ps1

# 手工最小复现
cmd /c "call E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat >nul 2>&1 `
  && E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -lineinfo -arch=sm_89 `
     -o tmp\outputs\D06\2026-10-08\sync_demo.exe `
     experiments\EXP-0006-synchronization\scripts\sync_demo.cu"
E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe -sass tmp\outputs\D06\2026-10-08\sync_demo.cubin
```

### 观察 1：SASS 里的屏障（`sass-key.txt`）

| 源码 | SASS（sm_89） |
|---|---|
| `__syncthreads()` | `BAR.SYNC.DEFER_BLOCKING 0x0` |
| `bar.sync 2,128` / `bar.sync 1,128` | `@P0 BAR.SYNC.DEFER_BLOCKING 0x2, 0x80` / `@!P0 BAR.SYNC.DEFER_BLOCKING 0x1, 0x80`（id=2/1，count=0x80=128） |
| `__syncwarp()` | `WARPSYNC 0xffffffff`（发散处还有 `BRA.CONV` 收敛分支；屏障前常插 `WARPSYNC`） |
| `__syncthreads_count/and/or(p)` | `BAR.RED.POPC/AND/OR.DEFER_BLOCKING 0x0, !P0 或 P0`（屏障与归约是同一条指令；谓词极性跟随上游 ISETP：条件 `threadIdx.x<16` 实测为 `!P0`，条件 `threadIdx.x%2` 实测为 `P0`） |

结论：`__syncthreads` 就是无 count 的 `bar.sync 0`；`__syncwarp` 就是 `WARPSYNC mask`；命名屏障在 SASS 里显式带 `id, count`。

> 末行（`BAR.RED.*`）由单独探针（`__syncthreads_count/and/or` 各一个 kernel）编译确认，未包含在 `sync_demo.cu` 里；谓词不是固定极性，`!P0` / `P0` 两种（随源码条件变化）都各自实测过。

### 观察 2：行为（`run-*.txt`）

| 模式 | 观察 |
|---|---|
| `divergent-syncthreads` | `FINISHED`（另一半 EXIT） |
| `divergent-syncthreads-alive` | `WEDGED — kernel still unfinished after 0.76–0.78 s`（看门狗实测打印） |
| `divergent-syncwarp` | `FINISHED` |
| `syncwarp-partial-alive` | `WEDGED — kernel still unfinished after 0.78 s`（看门狗实测打印） |
| `named-barrier` | `group0 correct 128/128`（两组各用 `bar.sync 1/2,128` 独立同步） |
| `smem-order-sync` | 0/256 错值（屏障建立 block 内可见性） |
| `smem-order-unsync` | 0/256 错值 —— **竞争"看起来正确"**，正是 racecheck 存在的理由 |

### 观察 3：ptxas 硬约束（`ptxas-badsync.txt`）

```text
$ nvcc -arch=sm_89 -cubin badsync.cu        # 内含 asm("bar.sync 1, 16;")
ptxas ... error : Number of threads participating in barrier must be in multiple of warp size
ptxas fatal   : Ptx assembly aborted due to errors
```

`count` 不是 32 的倍数 → 直接拒绝编译。对应 PTX ISA："When specifying a thread count, the value must be a multiple of the warp size."

### 观察 4：sanitizer 在本机被阻塞（诚实记录）

本机 `compute-sanitizer` **无法注入**，连一个只调 `cudaGetDeviceCount` 的 trivial 程序都报错：

```text
$ compute-sanitizer --tool racecheck sync_demo.exe smem-order-unsync
========= COMPUTE-SANITIZER
========= Error: process didn't terminate successfully
========= Error: Target application terminated before first instrumented API call
```

同机 `ncu` 能 attach，但计数器被拒：`ERR_NVGPUCTRPERM`。两者都需要**管理员权限 / 性能计数器权限**，而当前用户不是管理员（`IsAdmin=False`；`EnableDebuggerInterface` 注册表项已为 1）。所以 racecheck / synccheck 的"工具判词"只能留待有权限的环境复核（探针见 `_sanitizer-probe/`）。本节的所有正确性论断都用**行为观察 + SASS** 独立佐证，不依赖 sanitizer。

> 产物：`tmp/outputs/D06/2026-10-08/`（`sass-key.txt`、`run-*.txt`、`ptxas-badsync.txt`、`_sanitizer-probe/`）。

## 出处

- **CUDA C++ Programming Guide** → *"Thread Block Synchronization Functions"*（`__syncthreads()`）：
  > "waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to `__syncthreads()` are visible to all threads in the block." 且 "Allowed in conditional code only if the condition evaluates identically across the entire thread block."
- **CUDA C++ Programming Guide** → *"Warp Synchronization" / `__syncwarp()`*：
  > "Causes the executing thread to wait until all warp lanes named in `mask` have executed a `__syncwarp()` (with the same mask) before resuming execution… Each calling thread must have its own bit set in the mask… all non-exited threads named in `mask` must execute a corresponding `__syncwarp()` with the same mask, or the result is undefined."（sm_6x 另有"必须收敛且 mask 并集等于 active mask"的更严约束。）
- **CUDA C++ Programming Guide** → *"SIMT Architecture"*（Volta / Independent Thread Scheduling）：`__syncthreads()` 与 `bar.sync` 自 Volta 起**逐线程**执行，"will not succeed until reached by **all non-exited threads** in the block"。
- **PTX ISA** → *"Parallel Synchronization and Communication Instructions — `bar` / `barrier`"*：
  > "operand `b` specifies the number of threads participating in the barrier. If no thread count is specified, all threads in the CTA participate… the value must be a multiple of the warp size."；"`barrier{.cta}` instruction causes executing thread to wait for all **non-exited** threads from its warp and marks warps' arrival at barrier… `barrier{.cta}.sync` … wait for non-exited threads of all other warps participating in the barrier to arrive."；"Each CTA instance has sixteen barriers numbered 0..15."
- **CUDA C++ Programming Guide** → *"Cooperative Groups"*：`thread_block` / `tiled_partition` / `grid_group`；`grid.sync()` 需 `cudaLaunchCooperativeKernel`（`cudaDevAttrCooperativeLaunch`，CC 6.0+）。
- **PTX ISA** → *"Parallel Synchronization and Communication Instructions — mbarrier object"*：phase + pending count，`expect-tx`/`complete-tx` 跟踪异步事务，phase 完成触发。
- **CUDA C++ Programming Guide** → *"Asynchronous Barrier"*（`cuda::barrier` / `memcpy_async` 的 `cuda::pipeline`）。
- **libcu++（CUDA C++ Core Libraries）文档**：`cuda::barrier`、`cuda::latch`、`cuda::counting_semaphore`。

## 未决

- **`compute-sanitizer` 本机不可用**（非管理员 / 权限）。需要：以管理员运行，或配置 GPU 性能计数器/调试权限，再复核 `racecheck`（shared-memory hazard）与 `synccheck`（发散屏障 / barrier 误用）。当前结论靠行为 + SASS。
- 反例 A"EXIT 就不挂"是否可视为**实际安全**？文档层面仍是 UB（条件在整块上并不一致）。本稿按 **UB 处理**，不鼓励依赖。若确实只让一部分 lane 参与 warp 同步，正确写法是给 `__syncwarp` 传一个**明确的、恰好覆盖这些 lane 的 `mask`**——不要用 `__activemask()` 现算（它只反映调用瞬间的活跃 lane，不保证涵盖将来才会加入的线程，因此不能用来重新收敛）。
- **WEDGED 的可复现性**：本机观察到的是"看门狗轮询（实测 ~0.76–0.78 s）期间 kernel 仍未完成"，消息里打印的是**实测秒数**；若某台机器启用了 Windows 的 TDR（GPU 超时恢复，默认约 2s），同样的自旋内核可能被驱动直接掐断，此时 `cudaStreamQuery` 会返回错误、程序打印"kernel aborted"而非 WEDGED。两种表现都是"发散屏障得不到满足"，但脚本观察到的文字会不同。
- **命名屏障资源与 occupancy**：`tiled_partition` >32 会占用命名屏障，16 个上限如何影响 residency —— 待实测/查证。
- **`cuda::barrier` / `mbarrier`**：仅概念了解，尚未在本机跑最小例（sm_89 支持 mbarrier + `cp.async`，留待域 9）。
- **grid.sync()**：本机单卡可做 cooperative launch 最小验证，未跑。
- `__syncthreads()` 的 memory ordering 与 `acquire/release` 的精确边界（是否等价于 release+acquire，scope 为何）——与域 4 交叉，待合并校正。
