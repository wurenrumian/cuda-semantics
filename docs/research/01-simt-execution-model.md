---
title: 01 SIMT 执行模型
type: research
status: draft
created: 2026-10-08
updated: 2026-10-08
tags: [cuda, semantics, simt, warp, divergence, its, active-mask, sm_89, ada]
related: ["experiments/EXP-0001-simt-divergence", "docs/plan/cuda-semantics-topic-map.md"]
---

# 01 · SIMT 执行模型

> 目标：讲清 warp（32 线程）这个执行单位**承诺了什么、不承诺什么**。
> 环境：sm_89（Ada, RTX 4060 Laptop）/ CUDA 12.8 / nvcc V12.8.61。
> 所有观察来自本机实验，产物在 `tmp/outputs/D01/`，可重跑 `tmp/outputs/D01/run.ps1`。

## 一句话

**warp 是"调度与指令发射"的单位，不是"32 条线程一起执行"的承诺。**
同一时刻一个 warp 只发射一条公共指令，靠 **active mask** 决定哪些 lane 参与；发散的路径被**串行**执行。
Volta+/sm_89 的**独立线程调度（ITS）**让 lane 可以各自推进——但只推进到**下一个收敛点**为止。
于是，任何"隐式锁步（warp-synchronous）"的假设都是 UB。

## 承诺与边界

### 承诺（可以依赖的）

- **warp = 32 线程**，`warpSize == 32`，是 ISA 常量（本机与历代一致）。
- **一个 warp 一次执行一条公共指令**；不参与该指令的线程被置为 *inactive*（由 active mask 描述）。
  发散的 `if/else` 会**两条路径都执行**（先走一条、再走另一条），不是并行。
- **warp 级原语（`*_sync`）在给定 `mask` 下是良定义的**：`mask` 里每个自身位置有 bit 的线程，都必须执行**同一条、同一个 mask** 的原语，否则 UB。
  它们**不隐含内存屏障**，不保证内存序。
- **`__syncwarp(mask)` 保证参与线程之间的内存序**：写共享/全局内存 → `__syncwarp()` → 读其他参与线程写的值，是安全的。
- **`__activemask()` 返回"调用那一刻"活跃线程的快照**——这是它的全部承诺。

### 不承诺（想当然就会错的地方）

- 不承诺 warp 锁步：lane 不必在同一指令上。
- 不承诺发散后的**参与集合**：`__activemask()` 的返回值**不代表**后续指令的收敛集合。
- 不承诺 warp 级原语带来内存可见性（`__ballot/__shfl/__match/__reduce` 都"do not imply a memory barrier"）。
- 不承诺 lane 会/不会"跑到前面"：能跑多远由**收敛点（BSSY/BSYNC）**决定，不由语义规定。
- 不承诺 warp 内线程的执行顺序、延迟、发射顺序——这些是微架构行为（域 12）。
- 不能假定 UB 会以某种特定方式表现（比如"读到旧值"）：UB 是"结果无意义"；本机实测里它既可能看起来对（反例 C），也可能稳定错（反例 A）。

### 关键机制

- **active mask**：warp 当前指令的参与线程集合。硬件按它屏蔽 lane。
- **ITS（Volta+, CC≥7.0）**：per-thread PC + 调用栈，线程可在 sub-warp 粒度 diverge/reconverge；调度器把活跃线程重新组成 SIMT 单元。
- **收敛点**：结构化控制流的分支汇合处，编译器插入 `BSSY`/`BSYNC`（SASS 观察，见下）。早到的 lane 在此**等待**最慢的 lane。
- **SASS 对应**：`__activemask()`→`VOTE.ANY`；`__syncwarp()`→`WARPSYNC <mask>`；`__shfl_sync`→`SHFL.IDX`。

## 反例

> 每个反例都在本机跑过；命令与原始输出见「本机验证」。全部源码：`tmp/outputs/D01/src/d01_simt.cu`。

### 反例 A（核心）：拿 `__activemask()` 当 `__shfl_sync` 的 mask

**想当然**：`__activemask()` 返回"本 warp 的线程集合"，所以可以在发散分支里用它当 shuffle 的参与 mask。

```cpp
__global__ void k_shfl_bad(int *out) {
  int lane = threadIdx.x & 31;
  int v = lane * 100;
  if ((lane & 1) == 0) {                 // 只有偶数 lane 进入
    unsigned m = __activemask();         // 直觉=mask；实际=发散子集
    int x = __shfl_sync(m, v, lane + 1); // 想取奇数邻居 lane+1
    out[lane] = x;
  }
}
```

**实测**（`shfl-bad`）：进入分支的 16 个偶数 lane，`__activemask()` 返回 **`0x55555555`**（只有偶数位），
而 shuffle 的源 lane（奇数）不在 mask 里 → 结果**全错、且稳定**：16/16 都得到 `0`。

```
   0 : want  100  got      0  mask 0x55555555  <-- WRONG
   2 : want  300  got      0  mask 0x55555555  <-- WRONG
   ...                                          even-lane mismatches: 16/16
```

正确写法（`shfl-good`）：保持收敛，传满 mask，自己算源 index → `0/32` 错。

SASS 佐证（`sass-k_shfl_bad.txt`）：mask 由 `VOTE.ANY R5, PT, PT` 产生，shuffle 是 `SHFL.IDX PT, R7, R4, R0, 0x1f`，且整段在 `@!P0 EXIT`（奇数 lane 退出）之后执行。
**教训**：`mask` 必须由你**显式指定"逻辑参与者"**，不能用 `__activemask()` 反推。

### 反例 B：warp-synchronous 归约，只在循环前放一次 `__syncwarp()`

**想当然**：一次 `__syncwarp()` 建立屏障之后，后面的 warp 同步共享内存归约就安全了。

```cpp
__shared__ int red[32];
red[lane] = threadIdx.x;
__syncwarp();                                   // 只同步一次
for (int off = 16; off > 0; off >>= 1)
  red[lane] += red[lane + off];                 // 每一步都读别人正在写的槽
```

**实测**：`reduce-nosync` → `lane0=31`（正确应为 `0+…+31 = 496`），结果错。
把 `__syncwarp()` 放进循环每一步（`reduce-sync`）→ `lane0=496`，正确。

**教训**：跨 lane 的读-改-写**每一步**都需要同步，一次屏障不够。

### 反例 C：相邻 lane"写后读"依赖隐式锁步（UB 可能**看起来是对的**）

```cpp
__shared__ int sh[32];
sh[lane] = lane * 10;
int n = sh[lane ^ 1];   // 赌"邻居已经写完了"
```

**实测**：本机 `race-unsync` 打印出**逻辑正确**的邻居值（`10 0 30 20 …`），`race-sync` 也一样。
看似没问题——但这是**数据竞争（UB）**，不是保证。SASS 佐证：`sass-k_neighbor_unsync.txt` 里 `STS [R5], R0` 之后紧接 `LDS R7, [R4]`，**中间没有任何 `WARPSYNC`**。
**教训**：UB 的表现是"可能看起来对"，不能靠"跑很多次都对"来证明正确；这类旧代码必须补 `__syncwarp()`。

> 本想用 `compute-sanitizer --tool racecheck` 把 A/C 的竞争/掩码问题直接证出来，但**本机 sanitizer 注入失败**（见「未决」）。这里以行为 + SASS 为准。

### 反例 D（测量陷阱）：以为在发散循环后用 `clock64()` 能量出各 lane 的独立耗时

**想当然**：ITS 下 lane 各自推进，所以"工作量 ∝ lane"的循环里，小的 lane 应该更早读时钟、`t1-t0` 更小。

**实测**：`timing`（lane i 跑 `i*20000` 次串行 LCG，在 `if(lane>0)` 分支**内部**读时钟）：

```
   0 : 0
   1 : 5602628
   2 : 5602628
   ...              （lane 1..31 全部相同）
  31 : 5602628
```

lane1 的工作量是 lane31 的 1/31，耗时却相同。SASS 给了原因（`sass-k_timing.txt`）：时钟读 `CS2R R4, SR_CLOCKLO` 落在 `BSYNC B1` 之后——**循环出口是收敛点**，早完成的 lane 在 `BSYNC` 处等待最慢的 lane，然后所有 lane 一起读时钟。

**教训**：它不是"每个 lane 独立计时"，而是"warp 以最慢 lane 收尾"。想测量发散代价，要么在收敛点之前取值，要么换手段；**别把 `clock64()` 放在收敛点之后当作 per-lane 时间**。
（此点也提醒：ITS 的"独立推进"只在**收敛点之间**成立。）

## 本机验证（命令 + 观察）

### 复现命令

```powershell
# 1) 环境（必须先 vcvars64）
call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
# 2) 可执行 + cubin（用于 SASS）
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -lineinfo -arch=sm_89 -o bin\d01_simt.exe src\d01_simt.cu
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -arch=sm_89 -cubin -o bin\d01_simt.cubin src\d01_simt.cu
# 3) 行为
.\bin\d01_simt.exe mask | shfl-bad | shfl-good | reduce-nosync | reduce-sync | timing | intrinsics
# 4) SASS
E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe -sass -fun _Z6k_sassPii bin\d01_simt.cubin
# 5) 一键（含全部日志与 SASS 落盘）
pwsh -File tmp\outputs\D01\run.ps1
```

### 观察 1：`__activemask()` 随程序点变化（`logs/mask.txt`）

| 位置 | 观察到的 mask | 说明 |
|---|---|---|
| 分支前（收敛） | `0xffffffff` | 全 warp |
| then 段 `lane<16` | `0x0000ffff` | 只有活跃子集 |
| else 段 `lane>=16` | `0xffff0000` | 另一半 |
| join 之后 | `0xffffffff` | 又收敛 |
| 发散 while 第 1 轮（`lim>=1`） | `0xeeeeeeee` | 还在循环的 lane |
| 第 2 轮（`lim>=2`） | `0xcccccccc` | 集合收缩 |
| 第 3 轮（`lim>=3`） | `0x88888888` | 只剩 `lane%4==3` |
| 循环后 | `0xffffffff` | 收敛 |

**结论**：`__activemask()` 是"此刻活跃"的瞬时快照，会随程序点/分支/迭代变化 → 不能当逻辑 mask。

### 观察 2：warp 级原语（`logs/intrinsics.txt`）

- `__ballot_sync(0xffffffff, lane&1)` = `0xaaaaaaaa`（奇数位）。
- `__reduce_add_sync(0xffffffff, tid)` = `496`。
- `__match_any_sync(0xffffffff, lane>>4)`：lane 0–15 → `0x0000ffff`，lane 16–31 → `0xffff0000`。
- 共享内存归约（每步 `__syncwarp()`）lane0 = `496`。

### 观察 3：SASS（`logs/sass-*.txt`）

| 源 | SASS 关键片段 | 说明 |
|---|---|---|
| `__syncwarp()` | `BSYNC B2` … `WARPSYNC 0xffffffff` | 收敛 + warp 屏障 |
| `__activemask()` | `VOTE.ANY R5, PT, PT`（循环里 `VOTEU.ANY UR4, UPT, PT`） | 取活跃快照 |
| `__shfl_sync` | `SHFL.IDX PT, R7, R4, R0, 0x1f` | shuffle |
| 发散 if/else + 循环 | `BSSY B0/B1/B2/B3 …` / `BSYNC B0/B1/B2/B3` | 收敛屏障 |
| 无同步的邻居读 | `STS [R5], R0` → `LDS R7, [R4]`（无 `WARPSYNC`） | 竞争 |
| 发散循环后的时钟 | `BSYNC B1`（@0x510）→ `CS2R R4, SR_CLOCKLO`（@0x520） | 时钟在收敛点之后 |

### 环境旁证（重要）

- `ncu`（Nsight Compute 2025.1.0）能连接进程并运行目标，但收集指标报
  `ERR_NVGPUCTRPERM`（当前用户无 GPU 性能计数器权限；需管理员或 `RmProfilingAdminOnly=0`）。
- `compute-sanitizer 2025.1.0` 对本机**任何** CUDA 程序（含最小 `cudaMalloc` 例子）都报
  `Error: process didn't terminate successfully` / `Target application terminated before first instrumented API call`，且目标自身没有任何输出 → **注入失败**，`racecheck/synccheck/memcheck` 本轮均未跑成。

### 产物

- 源码：`tmp/outputs/D01/src/d01_simt.cu`
- 脚本：`tmp/outputs/D01/run.ps1`
- 日志（行为 + SASS + sanitizer 尝试）：`tmp/outputs/D01/logs/`
- 二进制/cubin/PTX：`tmp/outputs/D01/bin/`

## 出处

**CUDA C++ Programming Guide (v12.8)** — `docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/`

- **§4.1 SIMT Architecture**（`#simt-architecture`）：warp = 32；"A warp executes one common instruction at a time…"; active/inactive 线程定义；**Independent Thread Scheduling**（Volta+，per-thread PC/调用栈，sub-warp 粒度 diverge/reconverge）；"any warp-synchronous code (such as synchronization-free, intra-warp reductions) should be revisited"。
- **§4.2 Hardware Multithreading**：warp 调度器、`Wsize == 32`、寄存器/共享内存与驻留 warp 数。
- **§7.6 Synchronization Functions**：`__syncwarp(unsigned mask=0xffffffff)`；"guarantees memory ordering among threads participating in the barrier"；mask 命名外/未执行线程 → UB。
- **§7.19 Warp Vote Functions**：`__all_sync/__any_sync/__ballot_sync/__activemask()`；
  原文——`__activemask()` "Returns a 32-bit integer mask of all currently active threads…threads that are convergent at an `__activemask()` call are not guaranteed to be convergent at subsequent instructions"；且 "These intrinsics do not imply a memory barrier."
- **§7.20 Warp Match Functions**：`__match_any_sync/__match_all_sync`（CC≥7.x）。
- **§7.21 Warp Reduce Functions**：`__reduce_*_sync`。
- **§7.22 Warp Shuffle Functions**（`#warp-shuffle-functions`）：`__shfl_sync/__shfl_up_sync/__shfl_down_sync/__shfl_xor_sync`；mask 约定；§7.22.3.3 "Reduction across a warp"。
- **§8.4.2.1.1 Warp-Synchronous Code Pattern**（cooperative groups）：何谓 warp-synchronous。
- **§7.4.5 warpSize**。

**PTX ISA 8.7（CUDA 12.8）** — `docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/`

- **§3.2 Independent Thread Scheduling**：per-thread PC/调用栈、yield、sub-warp 粒度收敛。
- **§9.5 Divergence of Threads in Control Constructs**。
- **§9.7.9.6 `shfl.sync`**；**§9.7.13.1 `bar`/`barrier`**；**§9.7.13.2 `bar.warp.sync`**；**§9.7.13.4 `membar`/`fence`**；**§9.7.13.9 `vote.sync`**；**§9.7.13.10 `match.sync`**；**§9.7.13.11 `activemask`**；**§9.7.13.12 `redux.sync`**。

> 注：PTX ISA 的节号会随版本浮动（如 9.x 里同一条给到 §9.7.13.x / §9.7.9.6），以上为 CUDA 12.8 对应的 **PTX ISA 8.7** 编号。

## 未决

1. **compute-sanitizer 本机不可用**（最高优先）：注入失败，导致「反例 A/C」的竞争/掩码问题**没有**被工具直接证实。
   待查：是否需管理员 / `EnableDebuggerInterface` 已为 1 / 是否是 driver 596.49 与 sanitizer 2025.1.0 不兼容 / 安全软件拦截注入。
   逐条证据：`tmp/outputs/D01/logs/racecheck-*.txt`、`synccheck-*.txt`、`memcheck-*.txt`。
2. **`ncu` 指标权限**：`ERR_NVGPUCTRPERM`。若拿到管理员，可采 branch-efficiency / divergence 指标作为域 1/12 的量化旁证。
3. **`reduce-sync` 的 SASS 里没有逐迭代 `WARPSYNC`，但结果正确**（`sass-k_reduce_sync.txt`）。
   需要弄清：nvcc 是否因"完全收敛"把 `__syncwarp()` 优化掉、改由别的手段（执行序？）保证，这是否**可靠**（还是本例的巧合）。
   对照：`k_sass` 里同一写法确实生成了显式 `WARPSYNC`。
4. **可否构造一个非计时的 ITS 可观测演示**：例如早完成的 lane 在 `BSYNC` 之前先写一个标志，让别的 lane 读到——用来把"独立推进只到收敛点"变成可打印的证据。
5. **反例 A 修正**：`__activemask()` 的一个**正当**用途（cooperative groups "Discovery Pattern"）与本文陷阱的边界，留待域 6（同步原语）交叉。
6. 与 **EXP-0001** 的重复：本域已把"邻居读竞争 / ITS / `__syncwarp`"做实；EXP-0001（`experiments/EXP-0001-simt-divergence`）可作为同题的更早草稿，待合并或标注关系。
