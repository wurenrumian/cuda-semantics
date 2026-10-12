# CUDA Semantics 文档改进建议

> 这份文档记录阅读 `docs/research/01` 到 `12` 后积累的改进方向。
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

## 05 · 原子语义

- 明确本篇与 04 的分工：04 讲通用内存模型与同步关系；05 聚焦原子操作的 RMW、原子接口、scope/order 参数和 CAS 构造。开篇用一张关系图避免两章像重复讲同一件事。
- 将本篇涉及的原语按功能分类，而不是只围绕 `atomicAdd` 展开：RMW（add/sub/min/max/bitwise/exchange）、条件更新（CAS / compare-exchange）、原子 load/store、fence；标明哪些是传统 CUDA intrinsic，哪些是 `cuda::atomic` / `cuda::atomic_ref` 接口。
- 单独解释 `fetch_add`：原子加并返回旧值；对照 `atomicAdd` 的返回值与 `old = 10, new = 11` 的小例子，避免把它当成另一种内存顺序或同步机制。
- 以同一操作展示三轴独立性：操作种类（如 fetch_add）、scope（block/device/system）、order（relaxed/acquire/release 等）。指出 relaxed 放松的是顺序约束，不降低原子 RMW 对原子对象的不可分割性。
- 补齐各原语的返回值/成功条件/支持类型速查；尤其说明 CAS 失败时返回什么、compare-exchange 的 expected 参数如何更新，以及传统 intrinsic 与 C++ atomic API 的接口差别。
- 把 `atomicCAS` 循环拆成逐步图解：读旧值、计算候选值、尝试 CAS、失败则用新观察值重算；说明何时收敛、竞争下重试成本，以及 ABA 等通用 CAS 风险是否适用于所举场景。
- 修正并澄清 producer-consumer 示例：不要把“producer fence = release、consumer fence = acquire”写成可机械套用的等价关系；明确 fence 必须与原子通信及其顺序关系组成完整协议，优先给出可直接验证的 `cuda::atomic_ref` release-store/acquire-load 版本。
- 给每个反例标注它验证的维度：丢失更新（原子性）、relaxed flag（排序/同步）、负浮点 atomicMax（表示与比较）、浮点求和（数值可复现性）、跨 block 自旋（进度/调度）。
- 将 PTX/SASS 表格加上“本机 sm_89 观察，非跨架构承诺”标签，并避免仅凭 SASS 的 `STRONG` 字样推断 CUDA memory order；把未决项放进显眼的边界提示。
- 明确传统 intrinsic 的默认 scope/order、各后缀适用范围与类型/架构限制；避免把某个 API/类型的限制泛化到所有 atomic API。

## 06 · 同步原语

- 开篇用“是否等待参与者 / 是否建立内存顺序 / 覆盖范围”三问分类，分开执行屏障、fence、原子通信、异步等待与 host 同步。
- 为 `__syncwarp`、`__syncthreads`、命名 barrier、`grid.sync`、`__threadfence*`、release/acquire、stream/event/kernel 边界做统一对照，并注明参与者约束和可用条件。
- 用一个 block 内跨 warp producer-consumer 和一个跨 block 例子对比：屏障能解决谁等待，memory order 能解决什么可见性，二者何时需要同时出现。
- 对命名 barrier 明确 barrier ID、到达计数、参与线程一致性及复用规则；区分它与 `__syncthreads()` 的全 block 参与语义。
- 讲清跨 block 自旋锁/屏障的进度陷阱：内存协议正确不代表等待方能让出资源；给出 kernel 边界和 cooperative grid barrier 两种更易证明的替代方案。
- 对故意 wedge 的实验先给安全提示和流程图，解释 host 看门狗能保护实验程序但不是 CUDA kernel 内的取消机制。
- 将 SASS 屏障指令证据放在语义解释之后，明确指令映射依赖编译器和架构。

## 07 · PTX 指令级语义

- 先建立源码 → CUDA C++ 语义 → PTX 虚拟 ISA → SASS 目标 ISA 的链路，明确 PTX/SASS 分别能证明什么、不能证明什么。
- 将 scope、memory order、cache/volatile 语义分成独立旋钮，给出最小成对例子，避免从 `STRONG`、cache operator 或 `.volatile` 单独推断同步保证。
- 对 `const __restrict__` 反例解释它是调用方契约被违反导致优化器可合法假设别名不存在，不是 GPU 缓存“不刷新”的直接证据。
- 说明 `__ldg` 的适用条件、编译器如何选择只读通路，以及不同架构上指令被合并/消除的可能性。
- 对每个 PTX/SASS 片段标明目标架构、编译参数和观察范围；将机器码当作实现证据，不替代语言模型证明。
- 增加“怎样读反汇编”的短指南：定位 kernel、追踪 predicate/branch、区分被删除代码与未执行路径，并链接到对应源码行。

## 08 · 浮点与数值语义

- 用一页概念地图区分 IEEE 754 基本运算、FMA contraction、FTZ/denormal、近似数学、并行归约顺序与原子更新顺序。
- 每个反例同时给出“数学表达式、浮点逐步舍入、编译选项、实际指令/结果”，避免只看数值表而不知道差异来源。
- 解释 FMA 是一次舍入而非先乘后加两次舍入；用能区分两种结果的最小输入展示，并说明 `-fmad=false` 的作用边界。
- 讲 FTZ 时区分输入 subnormal、运算中间值、输出以及不同指令/路径；避免从某条 global atomic 的 FTZ 观察推广到所有浮点运算。
- 对 `--use_fast_math` 列出具体改变的选项/函数近似和精度代价，不笼统描述为“精度更低”。
- 将“同输入多次原子求和结果不同”与“单次浮点运算不符合规范”分开：前者通常是合法的不同归约顺序，强调误差界与确定性需求。
- 标注测量中时钟、编译器版本和架构条件；数值 bit pattern 结论尽量给可复现输入和 CPU/高精度参照。

## 09 · 异步与流水语义

- 开篇把 host API 异步、stream 排序、设备端异步 copy、流水重叠、graph capture 分开，说明“异步”不是单一保证。
- 给出 host、stream、kernel、copy engine、device thread 的时间线图；特别标明 API 返回、工作入队、工作开始、工作完成是不同事件。
- 用依赖图解释 event 的 record/wait：它建立 stream 间先后关系，不是全设备无差别同步；计时 event 与依赖 event 的使用也分开说明。
- 明确 `cudaMemcpyAsync` 是否真正异步取决于内存类型、方向、运行时和资源；区分 host 返回行为与 copy/kernel 是否重叠。
- 将 `cp.async` 的 issue、commit/wait、shared 可读时机及线程/warp 参与规则用状态图说明；避免把“异步搬运”误解成自动同步。
- 区分 CUDA stream 顺序、C++ host happens-before 与 device memory ordering，解释它们通过哪些 API/边界连接。
- 对 default stream 说明 legacy/per-thread 模式及编译/运行时配置对隐式同步的影响，避免“一定串行/一定独立”的绝对说法。
- 对 graph capture 标明 capture 阶段与 replay 阶段；指出捕获建立图，不代表捕获时 kernel 已执行。
- 对 TMA / `cp.async.bulk` 清楚标出 sm_90 前置条件，并用“本机不支持”区分架构限制与实验未完成。

## 10 · 宿主侧运行时与启动语义

- 用一次 kernel launch 的生命周期图串起配置、参数传递、异步提交、执行、错误产生与错误观察点。
- 将 launch configuration 中 grid/block/shared memory/stream 分别解释，补全维度计算和资源限制的直觉例子。
- 把即时 launch error 与异步执行 error 做成时间线，说明 `cudaGetLastError`、`cudaPeekAtLastError`、同步点各自会观察/清除什么。
- 用表格区分 `cudaDeviceSynchronize`、`cudaStreamSynchronize`、event wait、host API 返回；特别说明哪些会阻塞 host、哪些建立设备工作依赖。
- 对多 stream 例子画依赖 DAG，避免把“不同 stream”误说成保证并行，或把 kernel 边界误说成跨 stream 全局顺序。
- 解释 context/device/module 的归属与生命周期，补充多 GPU 场景下当前 device、指针归属、peer access 的边界。
- 给错误传播反例加上最小正确检查模板，包含 launch 后检查和最终同步处检查。

## 11 · 编译器行为与未定义行为

- 开篇区分语言/ CUDA 模型未定义行为、违反 API 契约、编译器优化结果和硬件偶发现象，避免把所有“结果怪”都叫 UB。
- 对 `__restrict__` 反例画出别名关系与优化前后 load/store 顺序，明确责任在调用方违约，不是优化器随机重排。
- 解释 `volatile` 保证访问语义的边界：不提供原子性、线程间 happens-before 或通用 device fence；按 host/device 语境分别说明。
- 将数据竞争判定条件拆解为冲突访问、不同线程、非原子且缺少同步；再说明某些特定 CUDA 访问可能有额外原子性规则，避免过度绝对化。
- 加入“如何判断优化导致的现象”：对照 `-O3`、`-G`、PTX、SASS 与规范；强调 `-G` 改变代码，不是正确性修复。
- 对 sanitizer 不可用明确标成环境限制，并说明因此哪些结论缺少独立动态检测，不能用失败日志替代语义论证。
- 将每个实验结论标注为规范结论、编译器观察、硬件观察或性能测量。

## 12 · 架构与微架构

- 开篇区分 ISA/编程模型保证、可查询架构属性、微架构实现细节、性能测量四层，并给后文每条结论加类别标识。
- 增加从 compute capability 到具体特性可用性的查询流程；区分“代码能为某架构编译”“设备支持”“运行时可用”。
- 对 shared memory 容量例子解释每 SM 与每 block 限制、静态与动态分配、opt-in 属性和 occupancy 的关系。
- 对跨架构 SASS 对照采用相同源码/选项并列展示，解释差异但不把某一代指令形态写成语义要求。
- 对访存性能例子补充访问模式、缓存状态、工作集、计时方法和重复次数；强调单机时延比值不可直接推广到其他 GPU。
- 解释架构特性门控与 PTX fallback/JIT 的区别，注明 `-arch`、`-code`、fatbin 内容如何影响最终执行目标。
- 补充“能查询到不等于能依赖”：设备属性和 profiler 数据用于选择策略，正确性仍应由 CUDA 语义保证。

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

1. 先改 01、04、05 的概念分工与术语分类：warp 原语、内存模型、原子 API 彼此链接但不重复堆叠。
2. 再改 02、06 的层次/同步路线图，用一致的“等待、顺序、scope、进度”框架解释同步。
3. 补 03、07 的内存落点与源码→PTX→SASS 阅读路径。
4. 最后统一 08–12 的证据标签、实验边界和跨章链接，降低长篇实验记录的阅读负担。
