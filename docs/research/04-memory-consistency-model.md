---
title: 内存一致性模型
type: research
status: draft
created: 2026-10-08
updated: 2026-10-11
tags: [cuda, memory-model, consistency, weak-ordering, fence, atomics, scope, volatile, sm_89, cuda-12.8]
related:
  - docs/plan/cuda-semantics-topic-map.md
  - INTENT.md
  - experiments/EXP-0004-memory-consistency-model/scripts/memmodel_demo.cu
---

# 域 04 — 内存一致性模型

> 面向 **sm_89 (Ada, RTX 4060 Laptop) / CUDA 12.8**。所有"本机观察"都在该环境实测，命令与产物见文末。
> 实验源：`experiments/EXP-0004-memory-consistency-model/scripts/memmodel_demo.cu`；产物：`tmp/outputs/D04/2026-10-08/`。

## 一句话

CUDA 内存模型是一个**弱序（weakly-ordered）模型**：默认的 `ld`/`st` 既**不承诺原子性**、也**不承诺跨线程顺序或可见性**；要建立 happens-before，必须显式地用**带作用域（scope）和顺序（order）的原子 / 栅栏**——而 `volatile` 不是同步原语。

## 承诺与边界

把每个访存/同步动作看成三根正交的轴，才不容易混：

| 轴 | 取值 | 管什么 |
|---|---|---|
| **强度**（atomicity / morally strong） | weak 访存 / 原子（RMW）/ `volatile` | 这次访问是不是"一次性、不可撕裂"的；并发同址是否被定义为 data race |
| **作用域**（scope） | `_thread` / `_block` / `_cluster` / `_device` / `_system` | 这次同步能让**哪些线程**观察到顺序效果（`block < device < system`） |
| **顺序**（order） | `relaxed < acquire / release < acq_rel < seq_cst` | 这次同步给前后的访存**排不排序、怎么排** |

**承诺：**

- **默认即弱序，且同址竞争是 UB。** "两个线程在无同步的情况下读写同一地址"没有"读到旧值"这种"良定义但旧"的结果——它是未定义行为，结果可以是任何东西（指南 §7.5 的 `X`/`Y` 例子）。所以视角不应是"会不会读到旧值"，而是"我的程序有没有数据竞争"。PTX 侧把这条形式化为"冲突 + 不可道德化（not morally strong）即 data race"（PTX ISA §8.7、§8.7.1）。
- **顺序与可见性是两件事。** 栅栏（fence）只**排序**，它自己**不"发布"**写。指南 §7.5 明说：fence 只影响本线程访存的顺序，**不保证这些访存对其他线程可见**；可见性要靠强访问 / 原子 / `volatile` 等等价手段。← 这是最反直觉、也最容易踩的一条。
- **作用域决定"能同步谁"。** 跨 block 用 `_block` 手段不够（`_block` 只覆盖同一个 block）；跨 device / 与 host 交互必须 `_system`。C++ `<atomic>` 没有"作用域"这一维（等价于全系统级），"作用域"是 CUDA 相对 C++ 内存模型独有的轴。
- **顺序是"配对"的。** 一个 release 写要和一个 acquire 读（或同作用域的 fence）配对，才产生 happens-before。单边（只有 fence、或只有 relaxed 原子）不构成同步。
- **cumulativity（传递可见性）。** 当 release/acquire 的 scope 覆盖了"另一个线程已经看到的写"时，该写也会被传递地发布出去；GPU 为此会保守地扩大 flush 范围——这也正是"内存域 / fence 干扰"性能问题的来源（指南 §3.2.7.1）。

**边界（明确不承诺的）：**

- 不承诺"程序顺序 = 观察顺序"：默认 `st.global` 之间、`ld.global` 之间、读写之间都允许重排。
- 不承诺"原子 = 同步"：默认的 `atomicAdd` 等是 **relaxed** 的（见反例 ②）。
- 不承诺 `volatile` 的原子性或顺序性：它只要求编译器**每次都真访存**（指南 §14.5.3.3），不产生任何 happens-before（见反例 ①）。
- 不承诺"在某块硬件上没观察出错"：模型给出的是**保证**，不是"本机没复现就安全"（见反例 ④）。
- 不承诺 `__syncthreads()` 之外的 block 间顺序；block 执行顺序、warp 顺序都不是承诺（见域 2）。
  更不承诺多个 block（或两个 grid）**同时驻留**——所以跨 block 的无界自旋可能永远等不到（见反例 ④）。

## 反例

全部是本机实测（sm_89 / CUDA 12.8），可复现（命令见下一节）。

### 反例 ①（核心）：`volatile` 不是同步原语——`volatile int` 自增会丢更新

直觉："`volatile` 让访问直达内存，多线程一起加就不会丢。" 错。`volatile` 只阻止编译器优化掉访问，**读-改-写仍是两步**，不是原子。

```cpp
__global__ void k_volatile_inc(volatile int* c, int iters) {
    for (int i = 0; i < iters; ++i) { int v = *c; *c = v + 1; }  // 非原子 RMW
}
```

`64 block × 256 thread × 4000 iter`：

```
mode=volatile grid=64 block=256 iters=4000 expected=65536000 observed=4820 lost=65531180   ← 丢了 99.99%+
mode=atomic   grid=64 block=256 iters=4000 expected=65536000 observed=65536000 lost=0        ← 对照组
```

（`observed` 随每次调度浮动：2026-10-08 记录 4820，2026-10-11 复测 4738——量级稳定在千级、
丢失比例恒定在 99.99%+，但**不是同一数字**；atomic 一侧 65536000/lost=0 每次都复现。）

SASS 直击要害：`volatile` 编译成**系统作用域的强访问**，但每次仍是独立的 `load;add;store`，没有任何原子性：

```
LDG.E.STRONG.SYS R5, [R2.64]      // volatile 读
IADD3            R5, R5, 0x1, RZ  // +1
STG.E.STRONG.SYS [R2.64], R5      // volatile 写   ← 与上面的读-改之间没有原子保证
```

对照的 `atomicAdd` 编译成单条原子：

```
RED.E.ADD.STRONG.GPU [R2.64], R5
```

### 反例 ②：原子 ≠ 顺序——默认 `atomicAdd` 是 relaxed

直觉："我用原子置了 flag，前面的数据写自然就发布出去了。" 错。默认 `atomicAdd` **没有释放语义**，与前面的普通写之间没有顺序。

```cpp
__global__ void k_prod_atomic_relaxed(int* data, int* flag) {
    *data = 42;
    atomicAdd(flag, 1);          // 直觉以为这是"发布"，其实只是 relaxed RMW
}
```

本机 SASS（`data` 与 `flag` 之间**没有任何 MEMBAR**）：

```
STG.E                 [R2.64], R9     // data：弱序普通写
RED.E.ADD.STRONG.GPU  [R4.64], R7     // flag：原子，但与上面的写无顺序
```

PTX 亦如此：`atom.global.add.u32 %r2, [%rd3], 1;`（无 `.release`）。要发布，得配 `__threadfence()` 或改用 release 存（下表最后一行为对照）。

### 反例 ③：`__threadfence()` 只排序、不发布

即使加了 fence，指南 §7.5 仍强调它**不保证可见性**：真正把写"送出去"的，是 `volatile` / 原子 / 强访问。指南给的"最后一个 block 归约"样例里，`result` 必须声明 `volatile`，`__threadfence()` 只负责**顺序**（`result` 的写先于 `count` 的自增）。

本机能把这条"拆开看"：`device` / `system` fence 的 SASS 里出现 `MEMBAR.*` 与 `CCTL.IVALL`（失效本线程 L1，使后续读不吃到旧值），但对端的"能不能看到"仍取决于访问本身强不强。**别把 fence 当成"flush 通道"**。

### 反例 ④：本机没出错，不代表模型给了保证

跨 block 消息传递（block0 写 `data=12345` 再置 `flag`，block1 自旋等 `flag` 再读 `data`）：

```
msg-naive  (仅 volatile)          iters=2000 received_flag=2000 violations=0 timeouts=0
msg-acqrel (release/acquire)      iters=2000 received_flag=2000 violations=0 timeouts=0
```

naive 版本 2000 次**一次都没翻车**——在本机上 L2 一致，两个 block 又恰好同时驻留。但它的 PTX 里 `data` 写和 `flag` 写之间**没有任何 fence**（`st.volatile` / `st.volatile`）：

```
st.volatile.global.u32 [%rd1], %r10   // data = 12345
st.volatile.global.u32 [%rd2], %r11   // flag = 1      ← 与上一行无顺序承诺
```

**"跑很多次都对"不是承诺，只是运气。** 换架构、换块调度、编译器换版本都可能改变观察。要保证，用 acq_rel 版（下一节）。

**别忘了还有「等不到」这一层。** 这里 `k_msg_*` 用 `<<<2,32>>>`：block0 生产、block1 消费。CUDA
**不保证**两个 block 同时驻留；本 kernel 只有 2 个小 block，实践中几乎总会同时驻留（所以
`timeouts=0`），但这**不是承诺**。万一 block1 先被调度、占住 SM 一直自旋，而 block0 排不上，就会
死锁。代码里给自旋加了上界 `spin_max`，把死锁降级成「报告 timeout」，但那只是让实验**不挂住**，
不是算法安全——跨 block 自旋要么保证驻留（如 cooperative launch + `grid.sync()`），要么别假设对端
会并行推进。

## 本机验证（命令 + 观察）

### 环境与构建

```bat
:: 在 experiments\EXP-0004-memory-consistency-model\scripts\ 下执行
call E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -lineinfo -arch=sm_89 -o memmodel_demo.exe memmodel_demo.cu
```

（`scripts\run.ps1` 一次跑完构建 / 运行 / `-ptx` / `-cubin`+`cuobjdump -sass` / `compute-sanitizer`，产物写入 `tmp/outputs/D04/2026-10-08/`。）

### 观察 1：运行期数字（上面反例 ①②④）

```
memmodel_demo.exe volatile-nonatomic 4000
memmodel_demo.exe atomic 4000
memmodel_demo.exe msg-naive 2000
memmodel_demo.exe msg-acqrel 2000
```

### 观察 2：**同一条生产者，六种写法的指令级差异**（本文最有用的一张表）

编译 `-cubin` + `cuobjdump -sass`，`k_prod_*` 都是"先写 `data=42`，再写 `flag=1`"：

| 写法（源码） | `data` 写 | 之间的**顺序**手段 | `flag` 写 |
|---|---|---|---|
| naive（裸写） | `STG.E` | **（无）** | `STG.E` |
| `__threadfence_block()` | `STG.E` | `MEMBAR.SC.CTA` | `STG.E.STRONG.SYS`† |
| `__threadfence()` | `STG.E` | `MEMBAR.SC.GPU` + `ERRBAR` + `CCTL.IVALL` | `STG.E.STRONG.SYS`† |
| `__threadfence_system()` | `STG.E` | `MEMBAR.SC.SYS` + `ERRBAR` + `CCTL.IVALL` | `STG.E.STRONG.SYS`† |
| `atomicAdd(flag,1)`（默认） | `STG.E` | **（无！）** | `RED.E.ADD.STRONG.GPU` |
| `__nv_atomic_store_n(…,RELEASE,DEVICE)` | `STG.E` | `MEMBAR.ALL.GPU` + `ERRBAR` | `ST.E.STRONG.GPU` |

† flag 在该变体里是 `volatile`，故编译为系统作用域强访问；这正是"scope 由访问/栅栏决定"的直观体现。

要点：
- **作用域就写在指令里**：`CTA < GPU < SYS`（`MEMBAR.SC.CTA/GPU/SYS`），跨 block 必须 `_device` 起。
- **默认原子确实没有顺序**（`RED` 那一行中间空白）——反例 ② 的机器证据。
- **release 存**是 `MEMBAR.ALL.GPU` + `ST.E.STRONG.GPU`；消费侧对应 `ld.acquire.gpu`（见下）。
- device/system fence 额外带 `ERRBAR` 与 `CCTL.IVALL`（失效本线程 L1），对应"fence 有实际代价"。

### 观察 3：正确的消息传递 PTX（release/acquire 配对）

正确写法（生产 `release` + 消费 `acquire`）在 PTX 层长这样：

```
// 生产者
st.volatile.global.u32 [%rd1], %r11      // data
st.release.gpu.b32     [%rd3], %r10      // flag：release 发布

// 消费者
$L:  ld.acquire.gpu.b32 %r7, [%rd3];     // flag：acquire 获取
     ... 读到 1 后 ...
     ld.volatile.global.u32 %r8, [%rd1]  // data
```

`st.release.gpu` / `ld.acquire.gpu` 是 PTX 层最直接的"三轴合一"：语义 = release/acquire，作用域 = gpu，强度 = 强访问。`__threadfence()` 等价于 `fence.sc.gpu`（nvcc 目前落成 `membar.gl`）；`__threadfence_block()` = `membar.cta`；`__threadfence_system()` = `membar.sys`。

### 观察 4：`compute-sanitizer` 当前**无法注入**（环境问题，待修）

本机实测：即便对一个 trivial kernel（`cudaMalloc` + 单指令 kernel），`compute-sanitizer --tool memcheck` 也失败：

```
========= COMPUTE-SANITIZER
========= Error: process didn't terminate successfully
========= Error: Target application terminated before first instrumented API call
```

按 NVIDIA 自己的文档排查：Windows 下需要注册表 `HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface = 1`——**本机已为 1**；`--target-processes all`、换目录、换调用方式都不行。结论：**本机 racecheck/memcheck 目前不可用**（不是本文程序的问题）。所以本文的结论不依赖 sanitizer，而用"运行期数字 + PTX/SASS"作证。域 6 / 域 11 若也依赖 sanitizer，需要注意同一阻塞。

## 出处

CUDA C++ Programming Guide（12.8）：

- **§7.5 Memory Fence Functions** — "weakly-ordered memory model"；`__threadfence_block/__threadfence/__threadfence_system` 各自的 scope（等价于 `cuda::atomic_thread_fence(seq_cst, thread_scope_{block,device,system})`）；**fence 只排序、不保证可见性**；"最后一个 block"归约样例（`volatile` + fence）。
- **§7.6 Synchronization Functions** — `__syncthreads()` 的 block 内嵌内存序；`__syncwarp()` 的 warp 内存序。
- **§7.14 Atomic Functions** — 原子清单；**§7.14.3** 的 `__nv_atomic_load/store` 系列（CUDA 12.8 新增，带 memory order + `__NV_THREAD_SCOPE_*`），是 C++ 内存模型在 CUDA 里的原生映射。
- **§14.5.3.3 Volatile Qualifier** — `volatile` 只禁止"把访问缓存进寄存器/L1"，不提供同步。
- **§3.2.7 Memory Synchronization Domains**（含 §3.2.7.1 Memory Fence Interference）— cumulativity、device vs system scope、fence 的保守 flush 与性能。
- **§9.2.2 Memory Model**（Dynamic Parallelism 章内）— 可作为"内存模型"术语的补充入口。

PTX ISA（v8.7，随 CUDA 12.8）：

- **§8 Memory Consistency Model** — §8.4.2 `volatile` Operation；§8.5 Scope（`.cta/.cluster/.gpu/.sys`）；§8.7 Morally strong operations、§8.7.1 Conflict and Data-races；§8.8 Release and Acquire Patterns；§8.9 Ordering of memory operations（program order / observation order / memory synchronization / causality）。
- **§9.7.10.8 `ld`**（`.acquire`/`.relaxed`/`.volatile`、`.weak`、cache op）、**§9.7.10.11 `st`**（`.release`/…）。
- **§9.7.15.4 `membar` / `fence`** — 含 `fence{.sem}{.scope}` 与 legacy `membar{.ss}`。
- **§9.7.15.5 `atom`**、**§9.7.15.6 `red`** — `.relaxed/.acquire/.release/.acq_rel` × `.cta/.gpu/.sys`。
- **§9.7.15.1 `bar` / `barrier`**、§9.7.15.2 `bar.warp.sync` — 屏障的内存序。

CUDA C++ Standard Library（libcu++）：

- Thread scopes / memory model：`thread_scope_{thread,block,cluster,device,system}`、`cuda::atomic`/`atomic_ref`、`cuda::atomic_thread_fence`、`cuda::barrier`。
- 指南 §7.5 明确把 `__threadfence*` 定义为对 `cuda::atomic_thread_fence(seq_cst, scope)` 的等价。

## 未决

- **`compute-sanitizer` 注入失败**（本机级，见观察 4）：需在真正的高权限 / 干净终端复现，或核对驱动 596.49 与 sanitizer 2025.1.0 的兼容；修复前 racecheck/synccheck 不可用。这是本项目域 4/6/11 的共同前置。
- **未在本机跑 `cuda::atomic_ref` 的完整 order 矩阵**（relaxed/acquire/release/acq_rel/seq_cst × block/device/system），只做了 inline-PTX `st.release.gpu` / `ld.acquire.gpu` 与 `__nv_atomic_*`。补这个矩阵能给出"每种 order 在 SASS 上的落点"。
- **system scope 只验到指令层**（`MEMBAR.SC.SYS` / `membar.sys`），未做真实的 device↔host 跨域发布实验（需 mapped pinned memory + host 端 spin 配合）。
- **cumulativity / decoupled look-back（CUB 的 acquire-mask 搜索）**未动手：这是"多生产者 + 单消费者"下最能暴露 cumulativity 与 scope 关系的实战模式，留作下一步。
- **未做 store-buffer 型 litmus**（两线程互写互读）在 `-O3` 下的编译器重排复现；本机 L2 一致 + 编译器保守，可能难复现，但值得用 PTX 反汇编确认是否存在重排空间。
- **`volatile` 的接入作用域**：观察到 nvcc 把 `volatile` 编译为 `STRONG.SYS`，但其**模型层**的精确表述（是否始终 system scope、与 `.mmio` 的关系）需回到 PTX §8.4 原文核对后再定稿。

---

## 附：本次运行产物

`tmp/outputs/D04/2026-10-08/`

| 文件 | 内容 |
|---|---|
| `volatile-nonatomic.txt` / `atomic.txt` | 反例 ① 的运行数字 |
| `msg-naive.txt` / `msg-acqrel.txt` | 反例 ④ 的运行数字 |
| `producers.txt` | 六种生产者变体的回读结果 |
| `memmodel_demo.ptx` | 全部 kernel 的 PTX（`st.global` / `membar.*` / `st.release.gpu` / `ld.acquire.gpu`） |
| `memmodel_demo.cubin` / `sass.txt` | `cuobjdump -sass`（`STG.E` / `MEMBAR.SC.*` / `RED.E.ADD` / `ST.E.STRONG` / `CCTL.IVALL`） |
| `racecheck-*.txt` | sanitizer 注入失败的证据（环境问题） |
| `build.log` / `ptx.log` / `cubin.log` | 编译日志 |

源码与 runner：`experiments/EXP-0004-memory-consistency-model/scripts/memmodel_demo.cu`、
`experiments/EXP-0004-memory-consistency-model/scripts/run.ps1`。
