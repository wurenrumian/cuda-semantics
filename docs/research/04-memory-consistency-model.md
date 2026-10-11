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

## 四个概念，用大白话各说一次

讨论内存一致性时最容易把四件事搅在一起。先把它们**拆成四个独立问题**：

- **原子性（atomicity）**：*这一次*访问能不能"整体完成，不被撕裂"？
  比如 64-bit 的写会不会被别人观察到"写了一半"。
  注意两个层次：**单次访问不撕裂**（64-bit 弱访问在未对齐/某些情况下可能被撕裂）
  和**读-改-写三步整体是一次原子事务**（atomicAdd 那种 RMW）是两码事——
  所以"普通 load/store"也不必说成"完全不原子"：对齐的 32-bit 访问在当前硬件上一般不会撕裂，
  但它没有"读改写一体"的原子性，也没有任何顺序保证。别把这两个层次混成一个词。
- **顺序（ordering）**：*我这边的*两次访问，别人会不会看到它们**换了次序**？（详见下文 order 标尺）
- **可见性（visibility）**：*我写的值*要等到什么时候、被什么操作之后，别人才"读得到"？
  注意 fence **只管顺序、不管可见性**——这是最反直觉的一条（见反例 ③）。
- **作用域（scope）**：以上三条承诺**向多大范围的人兑现**？只对本 block？整个 GPU？还是连 host 也算？
  （`block < device < system`。跨 block 至少要 device scope；GPU 与 host/CPU 交互必须 system scope。）

PTX 会说 "morally strong"。翻译成大白话：**两边访问必须"般配"看上去像同一次事务对等发生**——
同址、至少一方是原子的/强的、作用域覆盖对端。两条访问不般配 → 竞争（UB）。
术语本身不必背着，记住"**般配才不 race**"这个白话版就够了。

把这四件事放到一张三轴表里（第四件事"作用域"就是其中一根轴）：

| 轴 | 取值 | 管什么 |
|---|---|---|
| **强度**（atomicity / morally strong） | weak 访存 / 原子（RMW）/ `volatile` | 这次访问是不是"一次性、不可撕裂"的；并发同址是否被定义为 data race |
| **作用域**（scope） | `_thread` / `_block` / `_cluster` / `_device` / `_system` | 这次同步能让**哪些线程**观察到顺序效果（`block < device < system`） |
| **顺序**（order） | `relaxed < acquire / release < acq_rel < seq_cst` | 这次同步给前后的访存**排不排序、怎么排** |

**order 标尺用"通信场景"记，别当成抽象强度**：

| order | 场景 | 一句话 |
|---|---|---|
| `relaxed` | 计数器加一，只关心"不丢数"，不传递别的数据 | 只保证这一下不被撕裂，前后的事它不管 |
| `release`（写在发送方） | 我写完数据、最后按一下"传单已发"的按钮 | **按了这个按钮，按钮之前的所有写都得先到达** |
| `acquire`（读在接收方） | 我读"传单已发"到达后再读数据 | 读到 flag=1 之后读数据，**一定能读到 release 之前的写** |
| `acq_rel`（RMW） | 我既收（前面的 flag）也发（要给别人立 flag） | 一边 acquire 一边 release，收发都要 |
| `seq_cst` | 谁先谁后**全局只有一个真相**，适合复杂的双端协议 | 所有 seq_cst 操作排成一个全序，最贵 |

### producer/consumer 的完整闭环

把 release/acquire 的样子画出来（跨 block 消息传递的标准姿势）：

```text
生产者 block                          消费者 block
────────────                          ────────────
data = 42          ← 普通写
                 (release) flag = 1   ← 发布点：先于这行的一切写都"打包发送"
                                          |
                                          :  （flag 经 L2 可见）
                                          ↓
                                     读 flag（acquire）== 1   ← 获取点
                                     读 data == 42  ✓        ← 保证能读到 42
```

配对关系像两个钩子：release 和 acquire **必须各一端、且 scope 覆盖对方**，钩子才咬合；
只有单边（只 fence、或只 relaxed 原子）是空钩子，什么也没挂上。

## 承诺与边界

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
  还有一件事要分开：**内存协议写得对，不等于 kernel 一定能跑完**。就算你的 release/acquire 在纸面上无懈可击，
  跨 block 自旋仍可能因为调度/驻留问题永远等不到对端（对端 block 根本没被放上 SM），死锁照样发生。
  协议解决的是"看到什么"，不是"能不能活着看到"——后者是域 2 的驻留/调度问题，见反例 ④ 的讨论。

## 同步机制一览表（谁等人，谁排序）

一次把"等不等人"和"排不排序"分开（这张表要和域 2/6 对照着看）：

| 机制 | 作用域 | 需要双方到场？ | 建立内存顺序？ | 典型用途 |
|---|---|---|---|---|
| `__syncwarp(mask)` | warp | 是（mask 里的线程） | 是 | warp 内的写-读交接 |
| `__syncthreads()` | block | 是（全 block） | 是 | block 内跨 warp 协作 |
| `__threadfence_block()` | block | 否 | 排序 | block 内顺序约束（不打断执行） |
| `__threadfence()` | device | 否 | 排序 | device 范围顺序约束 |
| release/acquire 原子 | 随 scope | 否（配对即咬合） | 配对后是 | 跨 block / 跨 device 发布协议 |
| `grid.sync()`（cooperative） | 整个 grid | 是（全部 block） | 是 | kernel 内的全局汇合点 |
| kernel launch 边界 | grid / stream | 是（隐式） | 是 | 最常见的"安全停顿"，多阶段处理 |

读法：**"需要双方到场"的是屏障（等人），"只排序"的是 fence/原子（不等人）**。两者都"建立顺序"，
但代价和工作方式完全不同——屏障会让线程闲置，fence 只是给前后访存定序。

## 现代写法的最小示意

指南推荐的新风格是 `__nv_atomic_*`（C11 风格、带 scope 的内建）或 libcu++ 的 `cuda::atomic_ref`：
原子性、顺序、作用域**三件事一次写清**，不再依赖 `volatile` + `__threadfence()` 两件套：

```cpp
#include <cuda/atomic>

__global__ void consumer(cuda::atomic<int>* flag, int* data, int* out) {
    // acquire：读到 1 之后，能看到 release 侧之前的所有写
    while (flag->load(cuda::memory_order_acquire) != 1) {}
    *out = *data;                        // 保证读到 42
}

__global__ void producer(cuda::atomic<int>* flag, int* data) {
    *data = 42;
    // release：把上面这行"打包发送"，消费者 acquire 读到后即收到
    flag->store(1, cuda::memory_order_release);
}
```

也可以用内建函数版本（同样的语义，不用 include libcu++）：

```cpp
// data 的写 + RELEASE 语义的 flag 写，作用域 device
__nv_atomic_store_n(flag, 1, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE);
// 读取侧
int v = __nv_atomic_load_n(flag, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
```

与老式 `volatile` + `__threadfence()` 示例的差异：

- 旧风格需要两样东西**拼起来**：`volatile` 把访问变强 + fence 管顺序；两件套里任何一件漏掉都不安全。
- 新风格一个调用说清三件事（强度、顺序、作用域），编译器也更可能把它落到**单条**既有原子能力的
  SASS 指令上，而不是"若干访存 + 拦在中间的 MEMBAR"的组合。
- 反例 ①④ 对比演示过：`volatile int*` 自增会丢更新、`volatile` flag 协议"碰巧对"；换成
  `cuda::atomic` / `__nv_atomic_*` 后，原子性和顺序都是**语言承诺**，不再依赖运气和本机硬件脾气。

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

### 观察 3.5：order × scope 矩阵在 SASS 上的落点（2026-10-11 补）

`cuda::atomic<int, thread_scope_{device,system}>`（libcu++）下，"**写 data → 发布/获取 flag**"这一对动作的机器码
（实验：`experiments/EXP-0004-memory-consistency-model/scripts/order_matrix.cu` + `run_order_matrix.ps1`，
产物 `tmp/outputs/D04/2026-10-11/order-matrix/`，10 个组合全部编译运行通过）。生产者侧：

| order（生产侧） | scope=device | scope=system |
|---|---|---|
| `relaxed` store | `ST.E.STRONG.GPU`（**无**栅栏） | `ST.E.STRONG.SYS`（**无**栅栏） |
| `release` store | `MEMBAR.ALL.GPU` + `ERRBAR` → `ST.E.STRONG.GPU` | `MEMBAR.ALL.SYS` + `ERRBAR` → `ST.E.STRONG.SYS` |
| `acq_rel`（RMW fetch_add） | `MEMBAR.ALL.GPU` + `ERRBAR` → `ATOM.E.ADD.S32.STRONG.GPU` | `MEMBAR.ALL.SYS` + 同构 |
| `seq_cst` store | `MEMBAR.SC.GPU` + `ERRBAR` + `CCTL.IVALL` → `ST.E.STRONG.GPU` | `MEMBAR.SC.SYS` + 同构 |

消费者侧（flag load）：

| order（消费侧） | scope=device | scope=system |
|---|---|---|
| `relaxed` load | `LDG`（无附加） | 同构 |
| `acquire` load | `CCTL.IVALL`（失效**自己**的 L1 后再读） | 同构 |
| `seq_cst` load | `MEMBAR.SC.GPU` + `ERRBAR` + `CCTL.IVALL` | 同构 |

四个要点（都能挂回正文白话表）：

1. **"作用域写在指令里"在这里依然成立**：`GPU ≠ SYS` 落出的是同构指令换了后缀，机制完全对称。
2. **relaxed 原子 = 强访问但无顺序**：`ST.E.STRONG` 单条、原子、但没有任何 `MEMBAR`——
   正好印证反例 ② "默认原子只解决原子性"。顺带修正一个直觉：`relaxed` 的原子 store **不是**弱访问，
   它落地成 STRONG 单条指令；"弱"的是普通 `STG.E`。
3. **release/acq_rel 用 `MEMBAR.ALL.*`**（比 `__threadfence()` 的 `MEMBAR.SC.*` 更广：ALL 连自身读写也要保序）。
4. **acquire 在消费侧的实现是 `CCTL.IVALL`（失效本线程 L1 后再读）**，而不是栅栏——
   L1 不参与跨 block 一致性，本线程可能还缓存着旧内容；读 flag 前先失效自己的 L1，
   和 fence 携带 `CCTL.IVALL`（观察 2 里的用法）是同一个道理。
   每次 acquire 都要失效一遍自己的 L1，这就是 acquire 的实际代价（量化归域 12）。

### 观察 4：`compute-sanitizer` 当前**无法注入**（环境问题，待修）

> **2026-10-11 更新：已解决**。根因是 **Windhawk**（进程级 mod 注入平台，每个新进程被注入
> `windhawk.dll`）与 sanitizer 自身的 DLL 注入在 ntdll 撞车——停 Windhawk 服务后全部正常。
> 当时记录的"非管理员、注册表、驱动版本"等排查方向均为误判。详见过程记忆
> `tmp/docs/memory/windows-toolchain-pitfalls.md`。

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

- **（已解决，2026-10-11）`compute-sanitizer` 注入失败**（根因 Windhawk 注入冲突；见观察 4 上方更新与
  `tmp/docs/memory/windows-toolchain-pitfalls.md`）。停 Windhawk 后已补跑 `racecheck`
  volatile-nonatomic / atomic：**双方 0 hazards**——这本身是一个重要校准：**racecheck 只检测 shared memory
  hazard**，反例 ①④ 的丢更新/消息顺序是 **global memory** 上的竞争，racecheck 本来就看不见。
  global 侧竞争的证据仍以运行数字 + PTX/SASS 为准（本篇反例 ①②④ 不变）。
  产物 `tmp/outputs/sanitizer-2026-10-11/d04-racecheck-*.txt`。
- **（部分完成，2026-10-11）order × scope 矩阵**：device/system 两档全矩阵已实测（观察 3.5，
  实验 `scripts/order_matrix.cu`）。剩余：block/cluster scope、cross-order 的 RMW 语义细节
  （`fetch_or`/`CAS` 与 `fetch_add` 在 order 上的落点是否一致）。
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
