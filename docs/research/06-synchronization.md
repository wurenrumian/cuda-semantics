---
title: 域 06 —— 同步原语（__syncthreads / __syncwarp / 命名屏障）
type: research
status: draft
created: 2026-10-08
updated: 2026-10-12
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

## 先问三个问题，再挑原语

"同步"这个词在 CUDA 里混了五种机制，选型前先过三问：**等谁？建立什么顺序？覆盖多大范围？**

| 问题 | 回答它的是 |
|---|---|
| **要不要等**——等待方是否阻塞到"参与者到齐" | 执行屏障（`__syncthreads`/`__syncwarp`/命名屏障/`grid.sync`/`cuda::barrier`）会等；**fence、原子、event 依赖不"等完所有人"**——fence 只给本地操作定型，原子只在对象上排队 |
| **建立什么顺序**——只保屏障点前后的读写可见，还是建立完整的 happens-before | 屏障自带"屏障点"语义；fence + 原子组合（04 的 message-passing）给出一对操作之间的序；`cuda::barrier` 则是"屏障 + 显式 scope/order"的打包 |
| **覆盖多大范围**——warp / block / grid / 全系统 | warp 原语、CTA 原语、`grid.sync()`（cooperative launch）、host 侧 stream/event |

按这五族把全景摊开（右边标注后文小节）：

| 机制族 | 典型原语 | 等谁 | 建立什么 | 范围 |
|---|---|---|---|---|
| ① 执行屏障 | `__syncthreads`、`__syncwarp`、命名屏障 | 参与线程**都到达** | 屏障点前后访问对参与者可见 | warp / block |
| ② fence | `__threadfence_block/_device/_system` | **无人等待** | 把"之前"的读写压进作用域内所有观察者可看到的次序 | 按 scope |
| ③ 原子通信 | acquire/release 原子 load/store | 只在**原子对象**上排队 | 一对 release→acquire 的消息传递序 | 按 scope |
| ④ 异步等待 | `cuda::barrier`/`mbarrier`、`cudaEvent*`/stream wait | 到达/完成**事件** | 屏障+order 打包 / 工作依赖 | block / stream |
| ⑤ host 同步 | `cudaDeviceSynchronize`、`cudaStreamSynchronize` | host **阻塞**至设备工作完成 | kernel 边界（提交间的全局顺序点） | grid / stream |

两个最常见的组合拳，放在本域的定型认识里：

- **屏障管"等"，memory order 管"看"**——`__syncthreads()` 之后的可见性只覆盖 block；
  跨 block 的读写序要靠 ②+③ 协议（04 篇）。
- kernel 边界（提交完成）是 CUDA 里**最可靠的同步**：一个 grid 的写下一个 grid 全部可见——
  但它也是最粗的：建立的是"全 grid 全清"的全局顺序，代价和粒度不灵活（域 10）。

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

> 上表「编译成」一列是**本机 SASS 观察**（观察 1 定点核对），不是语义定义——换 arch/mark 可能变；
> 语义以左列的原语含义与右侧的"等待谁/可见性"为准。SASS 对照证据按"语义在先、指令佐证在后"
> 的次序放在 §本机验证，读表时别倒着用（从 `BAR.SYNC` 反推谓词参与规则是域 07 会专门拦的坑）。

### 屏障管"等"，序管"看"：两个最小场景

**场景 A（block 内、跨 warp 的生产者-消费者）**——一堵墙就够：

```cpp
__global__ void k_intra_block(float* buf, int n) {
  __shared__ float tile[256];
  // warp 0 是"生产者"：把输入装进 tile
  if (threadIdx.x < 128) tile[threadIdx.x] = compute(threadIdx.x);
  __syncthreads();                       // ① 等待：两半到齐
  // warp 1 是"消费者"：读 tile —— ② 可见性由同一个屏障覆盖
  float v = tile[threadIdx.x - 128];
  …
}
```

生产者写 `tile`、消费者读 `tile`，中间**只有一堵 `__syncthreads()`**：既是"等齐人手"，
又是"写对读者可见"——因为范围都在 block 内。**不需要任何 fence 或带 order 的原子**。

**场景 B（跨 block 的生产者-消费者）**——一堵墙不够，要"齐装二件套"：

```cpp
__global__ void k_producer(int* flag) {
  buf[0] = answer;                                   // ① 普通写（对别的 block 何时可见，无承诺）
  __threadfence();                                   // ② 排序：把上面的写"压"出去（device 范围）
  atomicExch(flag, 1);                               // ③ 原子立旗（relaxed + device）
}
__global__ void k_consumer(int* flag, int* out) {     // 需驻留/顺序保证，见 05 反例 1 的有界自旋
  while (bounded_spin()) ;                           // ④ 有界等待旗
  __threadfence();                                   // ⑤ acquire 侧：把旗的"读到"变成先后
  out[0] = buf[0];
}
```

为什么 B 需要"fence + 原子旗"而不是一堵墙：**跨 block 没有屏障原语**（除 `grid.sync()`），
等待只能自建（自旋/事件），而自旋读到的旗子本身不携带"buf 写完了"的顺序——所以等待（④）
与顺序（②⑤）分属机制 ③ 和 ②①；缺一件都是 04 篇反例的形状（丢消息/看不见）。
两个场景在同一份"生产者-消费者"词面下混合使用，是本域最常见的张冠李戴。

**何时两件同时要**：等待方在等"数据就叫到了"而不是"人到齐"时（B）；或等待本身建在
原子通信（消息标志）上时。纯粹"分阶段干活"（A）用屏障即可，加 fence 是画蛇添足——也加不了
跨 block 的任何保证。

### 命名屏障细则：id、count、一致性、复用

`bar.sync id, count`（内联 PTX 上一等公民，cooperative groups 的 tile >32 也 builds 在上）：

| 规则点 | 内容 | 违反的后果 |
|---|---|---|
| **barrier ID** | 每个 CTA 有 0..15 共 16 个命名屏障；**id 0 无 count 写法就是 `__syncthreads()`**（全 block 参与） | id 混用、双方用不同 id 到同一物理屏障 → 等不到，相互 wedge |
| **到达计数** | `count` = 期望到达的线程数，**必须 32 的倍数**（本机 ptxas 直接拒绝编译，见观察 3） | 少到、多到或非倍数 → 屏障永远不满足或编译失败 |
| **参与一致性** | 同一物理屏障的**所有参与者必须用相同的 (id, count)** 调用；这与"全 block 隐式参与"的 `__syncthreads()` 不同——`__syncthreads()` 不用协商 count，条件**在整块上一致**即合法 | 各自传不同值 → UB / wedge；这层"要不要协商"是它与 `__syncthreads()` 最本质的语义差别 |
| **复用** | 满足一次后该 (id) 屏障释放，可立刻复用；但若上一次还**没满足**，再发出第二次到达同一 id 的调用会混乱 | 典型如把 tile 划成多块**顺序**复用屏障时，天然安全；并发复用同 id 要额外协议 |

与 `__syncthreads()` 的一个对照记忆：后者**没有 count**——语义层要求"条件在整块上一致"（UB
边界见反例），把它从"人来协商"的负担变成"编译器/程序员保证一致"的负担；命名屏障把这一致性
变成**显式契约**（谁来、几个人、id 一致），换来了"子组各归各"的分治能力。

### 不承诺什么（边界）

1. **不承诺谁会到达**。你没让线程来，它就不来 —— 后果见下节。
2. **不承诺跨 block 的可见性**。`__syncthreads()` 之后，本 block 的 global 写对其他 block 仍可能不可见；要跨 block 需 `__threadfence()`（device scope）或带 `.gpu` scope 的原子/acquire-release。
3. **不承诺"没挂就是对的"**。发散调用是 UB，是否可见地挂掉取决于时序（见反例）。
4. **命名屏障 `count` 必须是 warp size 的倍数**，且必须恰好等于**会到达的线程数**；`id` 只有 0..15，用光会影响 occupancy（`tiled_partition` >32 时占命名屏障）。
5. **`__syncwarp(mask)` 的 mask 只能命名"已收敛/保证会到"的 lane**；每个调用线程自己那一位必须置位。默认 `0xffffffff` 在发散里不安全。
6. **屏障不产生跨 scope 的 release/acquire 语义**；需要跨观察者时配 fence/原子（域 4）。

## 反例（想当然会错）

> **先看安全边界再读实验**：变体 B 是**故意写成永不到达**的演示——整个闭环是
> 「kernel 卡住 → **host 线程轮询**（每 10 ms 打一次 `cudaStreamQuery`）→ 截止 0.5 s 判 wedge
> → 打印实测秒数并 `_exit(0)`」，外层 run.ps1 再套 30 s 硬超时。按时序读：
>
> ```
>   发散屏障 kernel 启动 ·一半 thread 永远到不了 BAR.SYNC
>            │
>            ▼
>   host 看门狗轮询 ──超 0.5s──▶ 判 WEDGED → 打印耗时 → 结束进程
>            │ 0.5s 内恢复(未发生)
>            ▼
>           正常完成
> ```
>
> 两个必须分开的层次：这个看门狗是**实验程序在 host 侧的外部保护**（它观察"kernel 没完"），
> **不是 CUDA kernel 内的取消机制**——CUDA 没有任何"从设备侧叫停 kernel"的语言原语，
> 卡住的 kernel 只能被 host 端（看门狗/`_exit`/驱动 TDR）从外面止住。任何"我在 kernel 里
> 加个超时就不会挂"的想法都属此列：设施在 host，kernel 自己停不下来。

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

### 观察 4：sanitizer（2026-10-12 更新：已解锁，拿到工具判词）

> **2026-10-11/12 更新**：此前"无法注入"的根因是 **Windhawk**（进程 mod 注入平台）与
> sanitizer 注入撞车（详见 00 篇「不承诺」与过程记忆
> `tmp/docs/memory/windows-toolchain-pitfalls.md`）；停 Windhawk 服务后全部工具正常。
> **重启后自启**，每次跑 sanitizer 前需再停（2026-10-12 的下列数据即重新停过一次才跑成的）。

解锁后在本域补跑（产物 `tmp/outputs/sanitizer-2026-10-12/d06-*`）：

```
compute-sanitizer --tool racecheck sync_demo.exe smem-order-unsync
  → Race reported between Write access at k_smem_order_unsync+0x80 ...（1 error hazard）
compute-sanitizer --tool racecheck sync_demo.exe smem-order-sync
  → RACECHECK SUMMARY: 0 hazards
compute-sanitizer --tool synccheck sync_demo.exe divergent-syncthreads
  → 程序照常 FINISHED；SYNCCHECK 无违规（ERROR SUMMARY: 0 errors）
compute-sanitizer --tool synccheck sync_demo.exe named-barrier
  → ERROR SUMMARY: 0 errors
```

三条**工具判词**都各自有教学价值：

1. **"竞争看起来正确"被 racecheck 打穿**——观察 2 里 `smem-order-unsync` 行为数据 0/256 错值，
   racecheck 却给出了 hazard（错误计数 1）。这正是"工具看内存访问形状，不看运行结果"的样板，
   眼见为实的 0 错值不能作为无竞争证明。
2. **synccheck 没揪住 A 变体是"应该的"**——A（EXIT）在运行时屏障**确实被满足**（另一半已退出、
   从计数摘除），没有可观察的违规；UB 危害在此派不出运行时证据。synccheck 是**运行时检测器**：
   只有真的有人带着不一致状态到达物理屏障才报警，"这段代码在别的分支布局下会翻车"属于
   语义层判断（本稿按 UB 处理）。
3. **synccheck 对 WEDGED 场景也给不出判词**——alive 变体套上 sanitizer 后仍是 wedge（看门狗照样
   兜底打印），挂着的线程在合法等待、没到的根本不在屏障点，无"违规"可报。死锁不是"栅栏违规",
   是"永不满足"——前者工具能抓，后者只能靠外部计时确认不死。

（以下为**解锁前**的历史记录，保留作现场存档：）

```
$ compute-sanitizer --tool racecheck sync_demo.exe smem-order-unsync
========= COMPUTE-SANITIZER
========= Error: process didn't terminate successfully
========= Error: Target application terminated before first instrumented API call
```

同机 `ncu` 能 attach，但计数器被拒：`ERR_NVGPUCTRPERM`（GPU 性能计数器权限，仍需管理员，
与 Windhawk 问题无关，见 00 篇）。当时竞争/可见性结论全部用**行为观察 + SASS** 独立佐证
——解锁后这些结论与工具判词**全部吻合**，佐证方法本身经受住了复核。

> 产物：`tmp/outputs/D06/2026-10-08/`（`sass-key.txt`、`run-*.txt`、`ptxas-badsync.txt`、`_sanitizer-probe/`；
> 解锁后补跑在 `tmp/outputs/sanitizer-2026-10-12/d06-*.txt`）。

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

- **（已解决，2026-10-12）`compute-sanitizer` 复核**：racecheck（smem-order-unsync hazard /
  sync 对照 0）、synccheck（无运行时违规报告，见观察 4 的两条"给不出判词"的校准）已补；行为 +
  SASS 的结论与工具判词吻合。**遗留**：synccheck 类工具对"UB 尚未发生"的分支布局（反例 A 形状）
  天生无感，语义层判断仍以规范原话为准。
- 反例 A"EXIT 就不挂"是否可视为**实际安全**？文档层面仍是 UB（条件在整块上并不一致）。本稿按 **UB 处理**，不鼓励依赖。若确实只让一部分 lane 参与 warp 同步，正确写法是给 `__syncwarp` 传一个**明确的、恰好覆盖这些 lane 的 `mask`**——不要用 `__activemask()` 现算（它只反映调用瞬间的活跃 lane，不保证涵盖将来才会加入的线程，因此不能用来重新收敛）。
- **WEDGED 的可复现性**：本机观察到的是"看门狗轮询（实测 ~0.76–0.78 s）期间 kernel 仍未完成"，消息里打印的是**实测秒数**；若某台机器启用了 Windows 的 TDR（GPU 超时恢复，默认约 2s），同样的自旋内核可能被驱动直接掐断，此时 `cudaStreamQuery` 会返回错误、程序打印"kernel aborted"而非 WEDGED。两种表现都是"发散屏障得不到满足"，但脚本观察到的文字会不同。（本次在 sanitizer 下复跑 alive 变体时实测：看门狗打印变成 ~5 s——插桩拖慢导致阈值后的轮询错过更多周期，也印证"实测秒数、非固定值"这一点。）
- **命名屏障资源与 occupancy**：`tiled_partition` >32 会占用命名屏障，16 个上限如何影响 residency —— 待实测/查证。
- **`cuda::barrier` / `mbarrier`**：仅概念了解，尚未在本机跑最小例（sm_89 支持 mbarrier + `cp.async`，留待域 9）。
- **grid.sync()**：本机单卡可做 cooperative launch 最小验证，未跑。
- `__syncthreads()` 的 memory ordering 与 `acquire/release` 的精确边界（是否等价于 release+acquire，scope 为何）——与域 4 交叉，待合并校正。
