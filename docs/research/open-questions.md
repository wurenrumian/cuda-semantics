---
title: 未决台账（各域「未决」汇总）
type: research
status: living
created: 2026-10-12
updated: 2026-10-12
tags: [cuda, semantics, open-questions, backlog, sm_89]
related:
  - docs/research/index.md
---

# 未决台账 — 各域「未决」汇总

> 本页是 `docs/research/00`–`13` 各篇原「未决」节的**原样汇总**。各篇正文已不再保留
> 「未决」节，正文里指向它的交叉引用改指本页（形如「未决台账 · 域 07 · 4」）。
>
> - 条目按域分组，顺序与各篇原文一致；编号供正文引用（如本页 `07-4` = 域 07 第 4 条）。
> - `（已解决…）` / `（已闭合…）` / `（已处理…）` 标注**原样保留**：表示该项已有结论，
>   仅作校准记录留档，不是待办。
> - 本页是**活页**，可随时增删；各域正文的"承诺与边界"结论不依赖它。

各域源文件：[00](./00-baseline) · [01](./01-simt-execution-model) · [02](./02-thread-hierarchy) · [03](./03-memory-spaces) · [04](./04-memory-consistency-model) · [05](./05-atomics) · [06](./06-synchronization) · [07](./07-ptx-instruction-level) · [08](./08-floating-point) · [09](./09-async-pipelines) · [10](./10-host-runtime) · [11](./11-compiler-and-ub) · [12](./12-architecture-microarchitecture) · [13](./13-primitive-reference)

---

## 00 · 环境基线

来源：[00-baseline.md](./00-baseline)

1. **admin 阻塞未解**：`compute-sanitizer` / `ncu` 需要管理员或 `EnableDebuggerInterface.bat` / `RmProfilingAdminOnly=0`。解掉后才能给域 04/06/11 补 racecheck / synccheck / initcheck 证据。
2. **驱动 ↔ toolkit 最低版本**：驱动 596.49 对 CUDA 12.8 的最低要求未逐条核对。
3. **环境探针的常态化**：已在 `experiments/EXP-0000-baseline/` 提供一键脚本；后续可考虑把它的输出存档作为「每次大改动前后」的基线快照。

## 01 · SIMT 执行模型

来源：[01-simt-execution-model.md](./01-simt-execution-model)

1. **（已解，2026-10-11）compute-sanitizer 本机不可用**：根因不是权限/注册表，而是 **Windhawk 进程注入冲突**（每个新进程被注入 `windhawk.dll`，与 sanitizer 注入在 ntdll 撞车；停 Windhawk 服务后恢复，详见过程记忆 `tmp/docs/memory/windows-toolchain-pitfalls.md`）。已补跑 `racecheck`：`race-unsync` → **RAW hazard ×128**（d01_simt.cu:76/77，反例 C 被工具证实）、`race-sync` → 0 hazards（对照），产物 `tmp/outputs/sanitizer-2026-10-11/d01-racecheck-*.txt`。遗留，反例 A 的 `shfl-bad` 掩码问题 racecheck 不查（它只管 shared memory hazard）；掩码类问题仍以"调用点形状 + SASS 无 `WARPSYNC`"为证。注意：Windhawk 重启会自启，跑 sanitizer 前需再停。
2. **`ncu` 指标权限**：`ERR_NVGPUCTRPERM`。若拿到管理员，可采 branch-efficiency / divergence 指标作为域 1/12 的量化旁证。
3. **`reduce-sync` 的 SASS 里没有逐迭代 `WARPSYNC`，但结果正确**。加 guard 后重新核对（`cuobjdump -sass -fun _Z13k_reduce_syncPi`）：SASS 中没有 `WARPSYNC`、`BSSY`、`BSYNC`，5 步被展开成带谓词的直线代码（`@!P0 LDS/STS`）。也就是说，nvcc 因为控制流完全收敛，把 `__syncwarp()` 消去了；步间顺序靠的是生成的代码，而不是编程模型的保证。待查：这种情况是否可靠（还是本例的巧合）。旧的 `tmp/outputs/D01/logs/sass-k_reduce_sync.txt` 是加 guard 之前的，尚未按新源重新生成。对照：`k_sass` 里同一写法确实生成了显式 `WARPSYNC`。
4. **可否构造一个非计时的 ITS 可观测演示**：例如早完成的 lane 在 `BSYNC` 之前先写一个标志，让别的 lane 读到——用来把"独立推进只到收敛点"变成可打印的证据。
5. **（已闭合，2026-10-11）`__activemask()` 的正当用途**：Discovery Pattern 已实测（「观察 4」；`activemask_good.cu`）——发散分支内"快照→分支屏障→稳定集合→集合内通信/源 lane 在集合内"合法；跨出发散边界的通信仍必须先收敛。与域 6 的交叉点（cooperative groups 如何用同一模式实现 `coalesced_active`）留待域 6 展开。
6. 与 **EXP-0001** 的关系：本域实验即 `experiments/EXP-0001-simt-execution-model/`（源 `d01_simt.cu`），"邻居读竞争 / ITS / `__syncwarp`"已做实，本文证据来自它。

## 02 · 线程层次与调度

来源：[02-thread-hierarchy.md](./02-thread-hierarchy)

1. **（已处理）旧的 `barrier` 死锁 demo 会挂住**：无界 `while (true) __nanosleep(1000)` 已移除，`barrier` 改成两个有界演示，`run.ps1` 也加了逐进程看门狗（见「观察 4」）。遗留问题：想用 `compute-sanitizer --tool synccheck` 直接抓"发散 `__syncthreads()`"，仍需要在 sanitizer 可用的环境里另写一个可被 sanitizer 收尾的 kernel。
2. **（已解决，2026-10-11）compute-sanitizer 未能取数**：根因是 **Windhawk 进程注入冲突**（非权限/注册表——详见过程记忆 `tmp/docs/memory/windows-toolchain-pitfalls.md`）。停 Windhawk 服务后已补跑：`synccheck barrier` + `memcheck sched` 均 **0 errors**，产物 `tmp/outputs/sanitizer-2026-10-11/d02-*.txt`。（注：本篇剩下的"发散 `__syncthreads()`"死锁 kernel 已从可跑模式移除，synccheck 只复核了现用作演示的有界 barrier 写法。）此坑的风险策略：Windhawk 重启自启，跑 sanitizer 前需再停；域 4/6/11 的公共前置已解除。
3. **`%smid` 虚拟化语义**：`SR_VIRTUALSMID` 是否跨 launch 稳定、与 `%nsmid`/物理 SM 的关系，未做实验。
4. **计算抢占**（Volta+ computer preemption）：block 被抢占后是否迁移、语义影响，未验证。
5. **多维 grid → warp 映射**：文档说 `threadIdx.x` 最快线性化，但"线性序 → warp/sub-partition"的具体映射未在本机观测（与域 12 交叉）。
6. **`ncu` 实测占用率**（scheduler/occupancy section）未跑；本稿用的是运行时 API `cudaOccupancyMaxActiveBlocksPerMultiprocessor`。
7. **cooperative groups `grid.sync()`**（唯一被承诺的跨 block 同步）留域 6；sm_90 的 cluster/DSMEM 本机不支持，仅概念覆盖。

## 03 · 内存空间语义

来源：[03-memory-spaces.md](./03-memory-spaces)

1. **（已解决，2026-10-11）local OOB 的 memcheck**：根因是 **Windhawk 进程注入冲突**（详见过程记忆 `tmp/docs/memory/windows-toolchain-pitfalls.md`；与驱动/权限无关）。停 Windhawk 后复跑 `memcheck --tool memcheck mem_spaces.exe oob` → **`Invalid __local__ read of size 4 bytes ... out of bounds`**（带行号与 host 回溯，正是 03 反例 1「local 是有界、每线程的真实内存区域」的直接证据）。产物 `tmp/outputs/sanitizer-2026-10-11/d03-memcheck-oob.txt`。
2. **ncu 计数器权限缺失**（`ERR_NVGPUCTRPERM`）。bank conflict（`l1tex__data_bank_conflicts_*`）、L1/TEX 命中需 `RmProfilingAdminOnly=0` 或管理员；本稿 bank conflict 只有 timing 级证据。
3. local vs shared 的延迟差距被循环开销稀释（1.18×）；更干净的测法（多条独立链 / 批处理）待补齐。
4. texture 只覆盖了 `__ldg` 只读通路；真实 **texture object / sampler** 路径（线性插值、边界模式等）未做。
5. register→local **spill 的触发阈值**（每线程寄存器上限、`__launch_bounds__`）归域 2 / 12；这里只指路。
6. L1 与 shared 的 **carveout**（`cudaFuncAttributePreferredSharedMemoryCarveout`）未测。

## 04 · 内存一致性模型

来源：[04-memory-consistency-model.md](./04-memory-consistency-model)

1. **（已解决，2026-10-11）`compute-sanitizer` 注入失败**（根因 Windhawk 注入冲突；见观察 4 上方更新与 `tmp/docs/memory/windows-toolchain-pitfalls.md`）。停 Windhawk 后已补跑 `racecheck` volatile-nonatomic / atomic：**双方 0 hazards**——这本身是一个重要校准：**racecheck 只检测 shared memory hazard**，反例 ①④ 的丢更新/消息顺序是 **global memory** 上的竞争，racecheck 本来就看不见。global 侧竞争的证据仍以运行数字 + PTX/SASS 为准（本篇反例 ①②④ 不变）。产物 `tmp/outputs/sanitizer-2026-10-11/d04-racecheck-*.txt`。
2. **（部分完成，2026-10-11）order × scope 矩阵**：device/system 两档全矩阵已实测（观察 3.5，实验 `scripts/order_matrix.cu`）。剩余：block/cluster scope、cross-order 的 RMW 语义细节（`fetch_or`/`CAS` 与 `fetch_add` 在 order 上的落点是否一致）。
3. **system scope 只验到指令层**（`MEMBAR.SC.SYS` / `membar.sys`），未做真实的 device↔host 跨域发布实验（需 mapped pinned memory + host 端 spin 配合）。
4. **cumulativity / decoupled look-back（CUB 的 acquire-mask 搜索）**未动手：这是"多生产者 + 单消费者"下最能暴露 cumulativity 与 scope 关系的实战模式，留作下一步。
5. **未做 store-buffer 型 litmus**（两线程互写互读）在 `-O3` 下的编译器重排复现；本机 L2 一致 + 编译器保守，可能难复现，但值得用 PTX 反汇编确认是否存在重排空间。
6. **`volatile` 的接入作用域**：观察到 nvcc 把 `volatile` 编译为 `STRONG.SYS`，但其**模型层**的精确表述（是否始终 system scope、与 `.mmio` 的关系）需回到 PTX §8.4 原文核对后再定稿。

## 05 · 原子语义

来源：[05-atomics.md](./05-atomics)

1. **（已解决，2026-10-11/12）compute-sanitizer 本机不可用**：根因是 Windhawk 进程注入冲突（见观察 5 更新）；本域已拿到 shared memory 竞争的 racecheck 证据。**遗留**：`publish` 是 global 广播竞争，racecheck 天生不查——它的独立判定仍缺工具层证据，见下一条。
2. **反例 1 未在本机复现**（0/2400）。要把它变成「跑得出错」的 litmus，需要跨 SM 的 store-buffer / message-passing 变体（域 4 的核心工具），本机尚未搭。当前判定依赖规范 + SASS/PTX。
3. **`publish` 的跨 grid 驻留问题未实测**：消费者已改为**有界自旋 + 超时报告**，本机已验证超时分支会正常返回（观察 4b）；但「consumer 先占住 SM、producer 永远启动不了」这一**真死锁**场景本机尚未构造（历史上 2400 轮全部在超时前收到旗子，本轮 200+200+2000 轮也是）。要复现，需要人为制造驻留压力（占满 SM 后再启动生产者）或换 cooperative launch 对照；这类实验会永远等，必须在外部计时守护下做。
4. **（已解决，2026-10-11，04 篇「观察 3.5」）`ATOM…STRONG.GPU` 里 `STRONG` 与 relaxed/acquire 的精确对应**：order×scope 矩阵已实测——本机 SASS 助记符的 `STRONG` 是硬件层属性、不与 memory order 一一对应（relaxed 原子同样 `STRONG`）；语义落点看随行的 `MEMBAR`/`CCTL.IVALL`。本篇观察 2/3 的通用标签②即为此结论。
5. **（已解决，2026-10-12，观察 6）`atomicAdd(float)` SASS 出现 `.FTZ`**：与 `-ftz` 开关无关的指令形态固定行为（PTX 层面即无 ftz qualifier）。仍是**本机 sm_89 观察**，其他架构未测。
6. **128-bit 原子（`.b128`）在 sm_90 的行为**未实测——本机无 sm_90，只证到「sm_89 编译失败」。
7. **浮点原子求和的可复现性**只在概念层（非结合），未做「同输入不同分块得不同低位」的最小实验（域 8 的位置）。
8. `_system` 原子的 system-scope 原子性附加条件（PTX §8.1.1）本机单卡不做实测。

## 06 · 同步原语

来源：[06-synchronization.md](./06-synchronization)

1. **（已解决，2026-10-12）`compute-sanitizer` 复核**：racecheck（smem-order-unsync hazard / sync 对照 0）、synccheck（无运行时违规报告，见观察 4 的两条"给不出判词"的校准）已补；行为 + SASS 的结论与工具判词吻合。**遗留**：synccheck 类工具对"UB 尚未发生"的分支布局（反例 A 形状）天生无感，语义层判断仍以规范原话为准。
2. 反例 A"EXIT 就不挂"是否可视为**实际安全**？文档层面仍是 UB（条件在整块上并不一致）。本稿按 **UB 处理**，不鼓励依赖。若确实只让一部分 lane 参与 warp 同步，正确写法是给 `__syncwarp` 传一个**明确的、恰好覆盖这些 lane 的 `mask`**——不要用 `__activemask()` 现算（它只反映调用瞬间的活跃 lane，不保证涵盖将来才会加入的线程，因此不能用来重新收敛）。
3. **WEDGED 的可复现性**：本机观察到的是"看门狗轮询（实测 ~0.76–0.78 s）期间 kernel 仍未完成"，消息里打印的是**实测秒数**；若某台机器启用了 Windows 的 TDR（GPU 超时恢复，默认约 2s），同样的自旋内核可能被驱动直接掐断，此时 `cudaStreamQuery` 会返回错误、程序打印"kernel aborted"而非 WEDGED。两种表现都是"发散屏障得不到满足"，但脚本观察到的文字会不同。（本次在 sanitizer 下复跑 alive 变体时实测：看门狗打印变成 ~5 s——插桩拖慢导致阈值后的轮询错过更多周期，也印证"实测秒数、非固定值"这一点。）
4. **命名屏障资源与 occupancy**：`tiled_partition` >32 会占用命名屏障，16 个上限如何影响 residency —— 待实测/查证。
5. **`cuda::barrier` / `mbarrier`**：仅概念了解，尚未在本机跑最小例（sm_89 支持 mbarrier + `cp.async`，留待域 9）。
6. **grid.sync()**：本机单卡可做 cooperative launch 最小验证，未跑。
7. `__syncthreads()` 的 memory ordering 与 `acquire/release` 的精确边界（是否等价于 release+acquire，scope 为何）——与域 4 交叉，待合并校正。

## 07 · 指令级 / PTX 语义

来源：[07-ptx-instruction-level.md](./07-ptx-instruction-level)

1. **SASS `STRONG.SM/GPU/SYS` 的精确语义未官方确认**。SASS 无公开 ISA 手册。目前只能推断它同时编码"缓存解析层级"（SM/L1、GPU/L2、SYS/DRAM）与某种序属性；`.ca→STRONG.SM`、`.cg→STRONG.GPU`、`.cv→STRONG.SYS` 与"解析层级"高度吻合，但**待用 `nvdisasm` 控制位或更多样本证实**。
2. **反例 1 无运行时工具判定（环境限制，如实标注）**：全局内存上的"丢失第二批 load"不属 racecheck 视野（racecheck 只查 shared memory hazard，04/05 已实测校准）；memcheck 也不管"语义读旧值"。2026-10-11 起 compute-sanitizer 本机已可用（根因 Windhawk 注入冲突，见 00 篇），但对本反例仍无对应工具——结论**只能**靠 SASS/PTX 人工判读，不能用"跑不出错"补位。
3. **cache operator 的实际缓存行为未用 `ncu` 验证**：`.cg` 是否真的绕过 L1、`.cs` 的 evict-first 命中率，需 L1/TEX 指标佐证（本轮只看了指令编码）。
4. **`ld.global.lu` 在 SASS 为 `LDG.E.LU`**，与 PTX 文档"全局地址上等价于 `.cs`"的说法表面不符——待确认是文档措辞、目标架构差异，还是 ptxas 保留 `.LU` 语义。
5. 与相邻域的边界：本文只到"**指令提供了哪些旋钮**"；**序的推理（happens-before、scope、fence 组合）**归域 04/05，**编译器重排与 UB** 的更多形态归域 11。本文的 `volatile`/`seq_cst` 观察与它们有意交叉，未去重。
6. 未展开：`.mmio`、`ld.relaxed`、`.el/.evict_*` 优先级提示、`cp.async` 路径（归域 09）。

## 08 · 浮点与数值语义

来源：[08-floating-point.md](./08-floating-point)

1. 只系统验证了 **float**。double 的 FMA 收缩 / `-fmad` 行为、以及 `__fma_rn` 未单独测。
2. **半精度 / bf16 / fp8 与 tensor core（wmma/mma）** 未在本机跑；sm_89 支持 fp8 mma，属域 12 的 fragment 布局问题，待补。
3. `-use_fast_math` 是否还触发更激进的**表达式重结合 / 向量化**，未逐项确认（本篇只覆盖收缩与 approx 替换）。
4. `atomicAdd(float)` 的 global-FTZ / shared-denormal 只做了**行为验证**，未看其 SASS（`RED`/`ATOM` 上的 FTZ 修饰）；`red`（不返回旧值）与 `atom` 是否同语义待查。
5. denormal 在 sm_89 上的**性能代价**未量化（属"性能语义"，会随架构变，见域 12）。
6. `__fdividef` / `__fsqrt_rn` 等其它 intrinsic 在 fast math 下的逐项 ULP 差异未补全。

## 09 · 异步与流水语义

来源：[09-async-pipelines.md](./09-async-pipelines)

1. **（已解决，2026-10-12）racecheck 对反例 C 的独立判定**：Windhawk 注入冲突解除后已补跑（见「关键观察」），racy 版 5 组 hazard、正确版 0。**遗留校准**：racecheck 能看见 async-proxy 的写是本次实测事实（`cp_async` 出现在写入方堆栈里），但"它对每类异步机制都有效"不应推广——工具对 proxy 的覆盖随版本/机制可能变化，未见例外情形时仍以 SASS `DEPBAR` + 行为数据兜底。
2. **`ncu` / `nsys` 未跑**：重叠率、copy engine 利用率、SM 占用等只用了墙钟时间；时间线证据待补（`ncu` 需计数器权限 `ERR_NVGPUCTRPERM`，与 Windhawk 无关）。
3. **统一内存未实测**：`cudaMallocManaged` / prefetch / 迁移语义待做（归域 10）。
4. **cp.async 只在单 stage 上验证**：多 stage double buffering 的 `wait_prior(N)` 语义、与 `__syncthreads()` 的组合，以及 `cp.async.ca` vs `cp.async.cg` 的差异尚未展开。
5. **Windows/WDDM 的并发上限**：`asyncEngineCount=1` 对 H2D/D2H 重叠的影响、WDDM 批处理对 stream 并发的限制，只做了定性观察。
6. 反例 E 的「铺满 GPU 则不再重叠」没做出对照实验，待补一个 grid 铺满版。

## 10 · 宿主侧运行时 / 启动语义

来源：[10-host-runtime.md](./10-host-runtime)

1. **（已解决，2026-10-12）compute-sanitizer**：Windhawk 注入冲突解除（见域 0）；本域已拿到 `deref_null` 的 memcheck 指令级判词（观察 10）。**遗留**：`memcheck` 报过的程序在 sanitizer 下的粘性表现（如错误后 kernel 继续跑/提前止步）未单独记录——反例 3 的粘性结论来自无插桩运行，两者不冲突但未交叉验证。
2. **并发度只做了逻辑证明，没测真重叠**：本域用自旋标志位证明了 default stream 的隐式串行，但没测"两个 non-blocking stream 是否真在同一时刻占用 SM"。要定量需 `nsys` 时间线（`ncu` 需计数器权限 `ERR_NVGPUCTRPERM`，与 Windhawk 无关，见域 0）。
3. **CUDA 12 runtime 模块级 API**（`cudaLibrary` / `cudaKernel` / `cudaLibraryLoadData`）未实测；本域只确认 runtime 不再暴露旧的 `CUmodule` 风格句柄。
4. **device-side graph launch / programmatic dependent launch (PDL)** 属于域 9 / 域 12 交界，本域未展开。
5. `cudaFuncSetAttribute` 对 occupancy 的连锁影响（域 2）只点到为止。

## 11 · 编译器行为与 UB

来源：[11-compiler-and-ub.md](./11-compiler-and-ub)

1. **（已解决，2026-10-12）compute-sanitizer（域 04/06/11 的公共阻塞）**：根因 Windhawk 注入冲突，停服即恢复。本域已拿到 racecheck（counter-shared 16 errors）与 initcheck（uninit 4 errors）判决。**遗留校准**：racecheck 对 global 竞争（counter-volatile 形状）不适用——仍以数值 + PTX/SASS + 语义定义叠加论证。
2. 未覆盖的 UB 点（本机可直接做，尚未做）：
   - 越界访存：03 域已拿到 `memcheck` 的 `Invalid __local__ read` 判词（同族证据，`tmp/outputs/sanitizer-2026-10-11/d03-*`）；本域的 global 越界读未单独跑。
   - 未初始化读：`read_uninit` 结果稳定为 `0 0 0 0`（本机 `cudaMalloc` 常给零页），**不能**据此认为"未初始化=0"；这一点已被 initcheck 判词补证（观察 D）。
   - 有符号整数溢出 / 移位 UB、`reinterpret_cast` 触发的严格别名、`memcpy` 之外的 type punning。
   - 编译器对"无副作用死循环"的处理（`while(1);` 是否被保留）。
3. `-restrict` 全局开关对所有 kernel 指针的影响未测；`-use_fast_math` 与 UB 的交互留给域 08。
4. `__restrict__` 触发 `ld.global.nc` 是本机（sm_89/CUDA 12.8）的代码生成选择；不同 ptxas 版本可能不同，语义结论（可重排）不变。

## 12 · 架构 / 微架构

来源：[12-architecture-microarchitecture.md](./12-architecture-microarchitecture)

1. **ncu 计数器被权限阻断（ERR_NVGPUCTRPERM）**：拿不到 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 等，合并访存只有墙钟证据。要让观察和文档的 32B sector 语义对上，需管理员开 GPU Performance Counters 权限（NVIDIA 控制面板 / `NVreg_RestrictProfilingToAdminUsers`）。
2. **sm_89a / sm_89f 的拒绝是 CUDA-12.8 的实测**，而且 12.8 对任何 `f` 后缀都拒绝（连 `sm_100f` 也是）——所以「出处」里引的 `compute_100f` / `compute_100a` 示例来自比 12.8 更新的在线版指南。至于更新版（如 12.9+）对 Ada 的族目标怎么定义、哪些 (arch, suffix) 组合合法，本机未验证。
3. **cubin vs PTX 的加载行为未直接跑**：sm_75 cubin 拿到 sm_89 上应因 "no kernel image" 失败、而 PTX 可 JIT——目前只有 doc 依据，未实测（可作为秒级小验证补上）。
4. **shared carveout（L1/shared 可配置划分）未探**：可用 `cudaFuncAttributePreferredSharedMemoryCarveout` 观察；本次只动了 shared 上限。
5. **SM 内 sub-partition 结构未直接观测**：scheduler 数、每 scheduler 的 warp 槽位、寄存器文件分区——需要 ncu 的 scheduler 指标（被权限阻断）。
6. **架构相关功能单元的吞吐**（SFU / DP / tensor core / `mma` fragment 布局）本次未触及，属域 08 与更细的微架构笔记。
7. **原子落点（L2 atomic unit）与 `red` / `atom` 吞吐**未直接测；域 05 的 `publish` 实验只覆盖正确性，不覆盖 L2 解析位置。

## 13 · CUDA 原语速查

来源：[13-primitive-reference.md](./13-primitive-reference)

1. 本篇的**函数名**取自本机头文件，但**一句话语义**是概括，权威定义仍在官方文档与对应域；若某条与域内正文冲突，以域内正文与 `出处` 为准。
2. `__v*` 视频指令族、`__nv_fp8_*`、texture/surface（`tex1Dfetch` 等）、`__nv_bfloat16_*` 等**未逐条展开**，只给代表与头文件指针。
3. sm_90 专属原语（`__cluster*`、`__u128Atomic*`、`__threadfence_cluster`、TMA）本机 **无**，仅标注，未实测。
4. 建议后续把本表与各域正文做一次**交叉核对**，把"一句话"里与域内表述不一致的条目收敛。
