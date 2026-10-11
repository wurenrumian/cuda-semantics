# CUDA Semantics 文档改进建议

> 这份文档记录目前阅读 `docs/research/01` 到 `04` 后积累的改进方向。
> 先记录建议，不直接改动研究正文，后续可按优先级逐篇落实。

## 总体原则

- 区分四个维度：语义作用域、物理位置、执行/同步范围、内存顺序。
- 区分“编程模型承诺”和“本机实测现象”；实测结果不自动等于可依赖保证。
- 区分真正的同步屏障、warp 协作原语、状态查询和编译器/硬件机制。
- 用“作用域 × 是否等待 × 是否建立内存顺序”解释同步原语。
- 先给直观心智模型，再给 PTX/SASS 证据，避免让指令级细节替代语义解释。

## 01 · SIMT 执行模型

- 增加边界声明：本篇主要讲 SIMT、ITS、divergence、active mask 和 warp 协作，不是 CUDA 同步原语总览。
- 将内容分成四层：执行与收敛、warp 协作/通信、warp 内存同步、更大作用域同步。
- 把 `__syncwarp()` 与 `__ballot_sync`、`__shfl_sync`、`__match_*_sync`、`__reduce_*_sync` 分开；后者不是通用内存屏障。
- 补齐并归类 `__all_sync()`、`__any_sync()`、四种 `__shfl_*_sync()`、`__match_all_sync()` 和 `__reduce_*_sync()`。
- 把 `__activemask()` 单独归为状态查询，重点解释它不能直接充当逻辑参与 mask。
- 增加 `__syncwarp()`、`__syncthreads()`、`grid.sync()` 的作用域对照表。
- 弱化“ITS 只推进到下一个收敛点”这类容易被理解成语言承诺的表述，说明具体调度顺序不属于程序语义保证。
- 分开解释编译器生成的 `BSSY/BSYNC` 与程序员显式写入的 `__syncwarp()`；自然控制流收敛不等于内存同步。
- 修正归约章节：`__syncwarp()` 只能解决阶段顺序/内存顺序，不能修复越界或同一阶段内的读写冲突。
- 说明编译器消除 `__syncwarp()` 不代表程序模型取消了同步保证；不能从某次 SASS 缺少 `WARPSYNC` 推断一般情况下不需要同步。

## 02 · 线程层次与调度

- 增加线程和 block 的线性编号公式：`threadIdx` 到 local tid、`blockIdx` 到 block id。
- 加入最小的 global thread id 示例，并强调逻辑编号不代表执行顺序。
- 将正文拆成四个主线：编号、block 调度、同步、资源/occupancy。
- 说明“block 作为生产者、其他 block 作为消费者”本身不是问题；问题是没有建立跨 block 的同步和内存顺序。
- 增加跨 block 同步方案表：kernel 边界、cooperative `grid.sync()`、atomic/fence 协议、cluster（适用架构）。
- 明确 `cudaDeviceSynchronize()` 是 host 等待，不是 kernel 内 device 线程之间的通用屏障。
- 强调 `occupancy != performance`：occupancy 是常驻能力，不等于吞吐量或速度。
- 补充动态 shared memory，以及静态 shared memory 与动态 shared memory 对 occupancy 的共同影响。
- 解释寄存器压力、spill 和 local memory 的性能代价。
- 把 `__syncthreads()` 与命名屏障 `bar.sync id,count` 的参与规则分开说明。
- 对“warp size 是 ISA 常量”“唯一跨 block 屏障”等过于绝对的表达加限定，区分当前硬件事实和语言层面承诺。

## 03 · 内存空间语义

- 保留“作用域、生命周期、物理落点、访问代价”四维框架，并在表格前先解释四者的区别。
- 增加“源码 → PTX state space → SASS → 硬件资源”的编译链示意，解释 `物理落点`、`PTX state space`、`典型 SASS` 三列。
- 用 shared、global、local 各给一个完整对照：CUDA 写法、物理位置、PTX 前缀、SASS 指令。
- 明确 local memory 是每线程逻辑私有，但通常使用 device memory；不是每线程独占一块片上内存。
- 强调 PTX 中出现 `.local` 不等于最终 SASS 一定有 `LDL/STL`；是否真正访问 local 要看最终指令和编译器优化。
- 增加“同一个 SM 不等于共享 shared memory”：shared 的隔离单位是 block，不是 warp 或 SM。
- 解释 L1、L2、device memory 的层级关系，并区分缓存共享与线程同步/内存可见性。
- 对 constant、texture/read-only 的描述注明访问通路、广播条件和架构/工具链差异。
- 说明 shared 不一定总比 global 快：bank conflict、访问模式和缓存命中会改变实际性能。

## 04 · 内存一致性模型

- 把 `atomicity`、`ordering`、`visibility`、`scope` 分成更直白的四个概念；“atomicity / morally strong”目前容易混淆。
- 增加 producer/consumer 的 release/acquire 图：数据写入 → release 发布 flag → acquire 读取 flag → 读取数据。
- 明确默认原子操作通常只解决原子性，不自动提供 release/acquire 顺序。
- 保留并强化“fence 只排序，不单独完成发布”的解释；fence 不是执行屏障，也不是自动把数据推送给其他线程。
- 增加 `__syncwarp()`、`__syncthreads()`、`__threadfence()`、release/acquire atomic、`grid.sync()` 的对照表。
- 分别解释 block、device、system scope，并明确跨 block 至少需要 device scope，GPU 与 host 交互需要 system scope。
- 将 `relaxed`、`release`、`acquire`、`acq_rel`、`seq_cst` 用通信场景解释，而不只列成强弱排序。
- 弱化“普通 load/store 完全不原子”的绝对表达，区分单次访问可能不可撕裂与读-改-写整体原子是两回事。
- 明确内存协议正确不等于 kernel 一定能完成；跨 block 自旋还可能因调度/驻留不足而死锁。
- 增加现代 `cuda::atomic_ref` 或 `__nv_atomic_store/load` 的最小示意，并注明其与旧式 `volatile + fence` 示例的差异。

## 建议的统一对照表

| 机制 | 作用域 | 是否等待其他线程 | 是否建立内存顺序 | 主要用途 |
|---|---|---:|---:|---|
| `__syncwarp(mask)` | warp | 是 | 是 | warp 内同步和内存通信 |
| `__syncthreads()` | block/CTA | 是 | 是 | block 内跨 warp 通信 |
| `__threadfence_block()` | block | 否 | 是 | block 范围内排序 |
| `__threadfence()` | device | 否 | 是 | device 范围内排序 |
| release/acquire atomic | 取决于 scope | 否 | 配对后是 | 发布/获取协议 |
| `grid.sync()` | cooperative grid | 是 | 是 | kernel 内跨 block 同步 |
| kernel launch 边界 | grid/stream | 是 | 是 | 多阶段 kernel |

## 推荐落实顺序

1. 先改 01 的概念分类和同步边界，避免后文继续混用“warp 原语”和“同步原语”。
2. 再改 04 的 producer/consumer、scope、order 解释，这是跨 block 通信的核心。
3. 补 02 的分层同步和 occupancy 边界。
4. 最后在 03 增加编译链和内存层级的图示，让物理落点、PTX、SASS 的关系更直观。
