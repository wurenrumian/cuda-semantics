---
title: 13 —— CUDA 原语速查：语义一句话
type: reference
status: draft
created: 2026-10-12
updated: 2026-10-12
tags: [cuda, semantics, intrinsics, reference, sm_89]
related:
  - docs/research/index.md
  - docs/plan/cuda-semantics-topic-map.md
  - docs/research/06-synchronization.md
---

# 13 —— CUDA 原语速查：语义一句话

> 环境：sm_89（Ada, RTX 4060 Laptop）/ CUDA 12.8（nvcc V12.8.61）/ driver 596.49。
> 本篇的函数名与类型，**从本机头文件批量抽出**（`E:\NVIDIA\CUDA\v12.8\include\`，含 `crt/`），
> 不是凭记忆列；标 🚫 的是本机（sm_89）不支持的 sm_90+ 原语。

## 一句话

这是一张**横向地图**：前 12 篇是逐域深挖，本篇把散在各域里的原语汇到一处，每个给**一句人话语义**——
它的作用只是"**先认脸、知道去哪一篇**"，**不替代**各域的承诺与边界；判定规则一律回到对应域。

## 怎么用这一篇

- 三个问题先问，再找原语：**要不要等？建不建顺序？覆盖多大范围？**（详见域 06）

  | 问题 | 谁负责 |
  |---|---|
  | **等**（阻塞到参与者到齐） | 屏障族（`__syncthreads` / `__syncwarp` / 命名屏障 / `grid.sync` / `cuda::barrier`） |
  | **排**（只给本地操作定型，不等人） | fence 与带 order 的原子读写 |
  | **范围** | warp / block / grid / system，看 scope |

- 每个原语都当**承诺**读：一句话是"它承诺什么"；"它不承诺什么"回到对应域。
- 一句话里出现"**不等**/**不排**/**只**"时，往往就是最容易想当然的地方，后文标了 ⚠。

## A. 执行层次与索引（内建变量，不是函数）

| 名字 | 一句话语义 | 备注 |
|---|---|---|
| `threadIdx` | 我在**本 block 内**的坐标 | 只读；随 launch 配置 |
| `blockIdx` | 我这个 block 在 **grid 内**的坐标 | 只读；block 间无执行顺序（域 02） |
| `blockDim` | 一个 block 有多少线程 | 只读 |
| `gridDim` | grid 有多少 block | 只读 |
| `warpSize` | warp 多大 = 32 | ISA 常量，跨架构固定（域 12） |
| `%laneid` / `%warpid` / `%smid` / `%clock64` | warp/lane/SM 编号、SM 时钟 | **特殊寄存器**，C 里没有名字，靠 inline PTX 读（域 07） |
| `__cluster*`（`__clusterIdx`/`__clusterDim`/…） | cluster 内的坐标与尺寸 | 🚫 sm_90 thread block cluster，本机无 |

## B. 屏障族（**等**：参与者到齐才继续）

| 原语 | 一句话语义 | 备注 |
|---|---|---|
| `__syncthreads()` | block 内所有**未退出**线程到达后放行 + 屏障点前后访问彼此可见 | block 级；隐含 block 作用域的内存序 |
| `__syncthreads_count(p)` / `_and(p)` / `_or(p)` | 同上，额外返回谓词的非零计数 / 全真 / 有真 | 省一趟投票 |
| `__syncwarp(mask)` | warp 内 `mask` 线程都到达后放行 | Volta+（含 sm_89）**必需的**真 warp 同步 |
| `__barrier_sync(id)` / `__barrier_sync_count(id, n)` | **命名屏障**：按 `id` 分组等 | 同 block 可有多个独立屏障 |
| `grid.sync()`（cooperative groups） | **跨 block** 全体到达后放行 | 需 cooperative launch；唯一跨 block 的"等" |
| `cuda::barrier` / `cuda::latch` / `cuda::counting_semaphore` | libcu++ 屏障族（可带 scope/order） | 比裸 PTX 更好拼装（域 06） |
| mbarrier（PTX / `cuda::barrier`） | **异步**屏障，配合 `cp.async` 做流水 | Ampere+；见域 09 |

> ⚠ **易错**：`__syncwarp` 不是"可有可无的旧习惯"。Volta 起线程**独立调度（ITS）**，warp 不再锁步；
> warp 内要通信/共享数据，就得显式 `__syncwarp`（域 01、12）。`__syncthreads` **不跨 block**。

## C. fence 族（**只排序，不等**）

| 原语 | 一句话语义 | 备注 |
|---|---|---|
| `__threadfence_block()` | 把本线程**此前**的读写排到 cta 作用域的观察者都认同的次序里 | 不针对任何对象 |
| `__threadfence()` | 同上，device 作用域（最常用） | PTX `membar.gl` → SASS `MEMBAR.SC.GPU` |
| `__threadfence_system()` | 同上，system 作用域 | 单卡下少见 |
| `cuda::atomic_thread_fence(order, scope)` | libcu++ fence：order/scope 都可传 | `cuda::atomic_signal_fence` 仅本线程内排序 |
| `__threadfence_cluster()` | cluster 作用域 fence | 🚫 sm_90，本机无 |

> ⚠ **易错**：fence **不"等"任何人**。它只约束**本线程**在它前后的内存操作，不让别的线程停下来。
> "排序"和"执行同步"是两件事（域 04、06）。

## D. 原子 RMW 族（不可分地改，**intrinsic**）

本机 `*atomic*.h` 实测存在的名字（每个都另有 `_block` / `_system` 变体）：

| 原语 | 一句话语义 | 备注 |
|---|---|---|
| `atomicAdd` | 原子地"读旧值 + 加"，返回**旧值** | 整型 32/64；`float`/`double` **只有 Add** |
| `atomicSub` | 原子地减，返回旧值 | 整型 |
| `atomicExch` | 原子地换入新值，返回旧值 | 整型 / `float` |
| `atomicMin` / `atomicMax` | 原子地取小 / 取大 | **整型才有**（无浮点 max/min！） |
| `atomicAnd` / `atomicOr` / `atomicXor` | 原子地位运算，返回旧值 | 整型 |
| `atomicInc(p, v)` | `old >= v ? 0 : old+1`（环绕加），返回旧值 | 环形计数 |
| `atomicDec(p, v)` | `(old == 0 \|\| old > v) ? v : old-1`（环绕减），返回旧值 | 环形计数 |
| `atomicCAS(p, cmp, val)` | 若 `*p == cmp` 则写 `val`，返回**当时** `*p` | **返回值不区分成败**；`== cmp` 即成功 |
| `__u128AtomicCAS` / `__u128AtomicExch` | 128-bit 条件更新 / 交换 | 🚫 sm_90+，本机无 |

> ⚠ **易错**：默认 intrinsic = **relaxed 顺序 + device 作用域**——**原子性**是有的，**内存序**是**没有**的；
> 原子**不是**同步原语、不建立 happens-before（域 05 反例 1）。`atomicCAS` 是唯一"万能积木"，
> 可自建任意 RMW。浮点 `Min/Max/Exch/CAS` **本机全没有**。

## E. 原子族（**libcu++**：scope/order 可传）

| 接口 | 一句话语义 | 备注 |
|---|---|---|
| `cuda::atomic<T, S>` / `cuda::atomic_ref<T, S>` | 拥有 / 借用一块存储来**原子**地读写 | 后者包住已有变量 |
| `.load(order)` / `.store(v, order)` | 单次**原子**读 / 写，**可携带内存序** | 老 intrinsic 无对应物 |
| `.fetch_add/sub/and/or/xor/max/min(v, order)` | 原子 RMW，返回**旧值** | 对应 `atomicAdd` 等，但可传 order |
| `.exchange(v, order)` | 原子交换 | 对应 `atomicExch` |
| `.compare_exchange_weak/strong(exp, des, …)` | 相等才写；返回 **bool**，失败时**自动更新 `exp`** | 对应 `atomicCAS`，但给结论 |
| `cuda::atomic_thread_fence(order, scope)` | 带 order/scope 的 fence | |
| `cuda::memory_order_relaxed…seq_cst`、`cuda::thread_scope_block/device/system` | order / scope 的名字 | **scope=模板参数，order=函数参数** |

> ⚠ **易错**：别忘了它和 intrinsic 是**两套语法、同一套语义**（域 05）；写成 `store(1, release)` /
> `load(acquire)` 才是"语义正确"的消息传递，比"relaxed 原子 + 旁挂 fence"更不容易漏（域 04、05）。

## F. warp 级原语（warp 内通信 / 投票 / 归约）

| 原语 | 一句话语义 | 备注 |
|---|---|---|
| `__shfl_sync(mask, v, src)` | 取 **`src` 号 lane** 的 `v` | 选的是 **lane**，不是寄存器（域 01） |
| `__shfl_up_sync` / `__shfl_down_sync(mask, v, d)` | 取 `lane∓d` 的 `v` | `_down` = `lane+d` |
| `__shfl_xor_sync(mask, v, m)` | 取 `lane^m` 的 `v`（蝶形） | 求归约常用 |
| `__ballot_sync(mask, pred)` | 返回 32-bit 掩码：谁 `pred` 真 | |
| `__any_sync` / `__all_sync(mask, pred)` | 非零表示"有 / 全"为真 | |
| `__match_any_sync(mask, v)` | 返回与本线程 `v` **相同**的 lane 集合 | |
| `__match_all_sync(mask, v, &pred)` | 全员相同则 `pred` 真并返回 mask | |
| `__reduce_add/min/max/and/or/xor_sync(mask, v)` | 在 mask 内归约 `v`，结果给**所有参与者** | sm_80+ |
| `__activemask()` | 返回**此刻**收敛的 lane 集合 | 语义窄，见下 ⚠ |
| `__shfl` / `__shfl_up/down/xor` / `__ballot`（无 mask） | pre-Volta 旧式用法 | 在 ITS 下**不安全**，别再用 |

> ⚠ **易错一**：这些原语**不隐含内存屏障**（官方原话 "do not imply a memory barrier"）——它们搬数据/投票，
> 不建立可见性（域 01）。
> ⚠ **易错二**：mask 必须是**真正参与**的线程。拿 `__activemask()` 当 mask，在发散分支里只会拿到"此刻还在一起"
> 的 lane，源 lane 可能不在集合里 → 结果直接错（域 01 反例 A）。

## G. 访存与 cache operator

| 原语 | 一句话语义 | 备注 |
|---|---|---|
| `__ldg(p)` | 走**只读数据路径**的 load | 常配 `const __restrict__` |
| `__ldca` / `__ldcg` / `__ldcs` / `__ldlu` / `__ldcv` | 带 cache 提示的 load（cache-all/global/streaming/last-use/volatile） | 只改**缓存行为** |
| `__stwb` / `__stcg` / `__stcs` / `__stwt` | 带 cache 提示的 store（write-back/global/streaming/write-through） | 只改**缓存行为** |

> ⚠ **易错**：cache operator **不是同步原语**——它改变数据**走哪个缓存**，**不**改变可见性/顺序承诺。
> 要顺序请用 fence 或带 order 的原子读写（域 04、07）。

## H. 位 / 整型 / 地址工具

| 原语 | 一句话语义 |
|---|---|
| `__clz` / `__clzll` | 前导零个数 |
| `__ffs` / `__ffsll` | 最低位 1 的位置（从 1 计） |
| `__popc` / `__popcll` | 置位数（popcount） |
| `__brev` / `__brevll` | 位反转（bit reverse） |
| `__byte_perm(a, b, s)` | 按选择子 `s` 从 `a,b` 里各取字节拼装 |
| `__funnelshift_l/lc/r/rc(lo, hi, sh)` | 把 `hi:lo` 当 64 位做漏斗移位，取 32 位 |
| `__mulhi` / `__umulhi` / `__mul64hi` / `__umul64hi` | 取乘积的**高位** |
| `__mul24` / `__umul24` | 24 位乘法（旧式，少用） |
| `__sad` / `__usad` | 绝对差之和（SAD） |
| `__hadd` / `__rhadd` / `__uhadd` / `__urhadd` | 半字加（普通/四舍五入，有/无符号） |
| `__dp4a` / `__dp2a` | 4 项 / 2 项点积累加（int8/int4），sm_61+ |
| `__isGlobal` / `__isShared` / `__isLocal` / `__isConstant` / `__isGridConstant` | 指针**落在哪个地址空间**的判断 |
| `__cvta_generic_to_*` / `__cvta_*_to_generic` | generic 地址与各具体地址空间**互转** |
| `__vadd2/4`、`__vmaxs2/4`、`__vabs2/4`、…（`__v*` 一族） | 打包短整型的 **SIMD 视频指令**（按 2/4 路并行） |

> 说明：`__v*` 与 `__vi*` 视频指令是**一整批**（`__vadd/__vsub/__vmin/__vmax/__vabsdiff/__vavg/__vcmp*/__vset*/…`，
> 各带 `s/u`、`2/4` 后缀），本机头文件都有；此处只列代表，完整清单见 `crt/device_functions.h`。

## I. 浮点与数值

**确定性舍入**（后缀 `_rn`（就近）/`_rz`（向零）/`_ru`（向上）/`_rd`（向下））：

| 原语 | 一句话语义 |
|---|---|
| `__fadd_rn/_rz/_ru/_rd`、`__fsub_*`、`__fmul_*`、`__fdiv_*` | 单精度四则，**明确指定舍入**（不被收缩/合并） |
| `__fmaf_rn/_rz/_ru/_rd`、`__fmaf_ieee_*` | 单精度**融合乘加**（`a*b+c` 一次舍入） |
| `__frcp_rn/_rz/_ru/_rd`、`__fsqrt_*`、`__frsqrt_rn` | 倒数 / 平方根 / 平方根倒数 |
| `__dadd_*` / `__dsub_*` / `__dmul_*` / `__ddiv_*` / `__drcp_*` / `__dsqrt_*`、`__fma_*` | 双精度对应物 |
| `__saturatef(x)` | 夹到 `[0.0, 1.0]` |

**快速近似 / 低精度**（“fast math” 家族，牺牲精度换速度）：

| 原语 | 一句话语义 |
|---|---|
| `__sinf` / `__cosf` / `__tanf` / `__sincosf` | 快速三角函数（精度低于 `sinf`） |
| `__expf` / `__exp10f` / `__logf` / `__log2f` / `__log10f` / `__powf` | 快速指数 / 对数 / 幂 |
| `__fdividef` | 快速除法 |
| `__tanhf` | 双曲正切 |

**类型转换 / 位重解释**：

| 原语 | 一句话语义 |
|---|---|
| `__float2int_rn`、`__int2float_*`、`__float2ll_*`、`__float2uint_*`、`__float2ull_*` | 浮点↔整型，带舍入后缀 |
| `__double2int_*`、`__double2ll_*`、`__double2float_*`、`__ll2double_*`、`__int2double_rn`、`__uint2double_rn`、`__ull2double_*` | 双精度↔整型/单精度 |
| `__float_as_int` / `__int_as_float` / `__float_as_uint` / `__uint_as_float` | **不改位**，只换解释（bit-cast） |
| `__double_as_longlong` / `__longlong_as_double`、`__double2hiint` / `__double2loint` / `__hiloint2double` | 双精度的位视角 |

> ⚠ **易错**：`-use_fast_math` 会**改变数值结果**（`--ftz --prec-div=false --prec-sqrt=false --fmad=true`）；
> FMAD 收缩默认开，同一个表达式加不加标志可能差位（域 08）。`__sinf` 不等于 `sinf`。

## J. 半精度 / bf16

| 原语 | 一句话语义 |
|---|---|
| `__float2half` / `__half2float`、`__float2half_rn/_rz/_ru/_rd` | 单精度↔半精度 |
| `__hadd` / `__hsub` / `__hmul` / `__hdiv`（及其 `_sat` 变体） | 半精度四则（`_sat` 结果夹到 `[0,1]`） |
| `__hfma` / `__hfma_sat` / `__hfma_relu` | 半精度融合乘加（`relu` = 负值归零） |
| `__hadd2` / `__hmul2` / `__hfma2` / `__hadd2_sat` / `__hmul2_sat` / `__hfma2_sat` | **`half2` 打包** SIMD 版本（一次算两个） |
| `__hmin` / `__hmax` / `__hmin_nan` / `__hmax_nan`（及 `…2`） | 半精度取小/取大（`_nan` 版有 NaN 传播规则） |
| `__heq` / `__hne` / `__hlt` / `__hle` / `__hgt` / `__hge`（及 `u`、`2`、`_mask` 变体） | 半精度比较（`u` = 无序） |
| `__habs` / `__hneg`（及 `…2`）、`__h2exp` / `__h2log` / `__h2sqrt` / `__h2rcp` | 绝对值/取负、half2 数学 |
| `__low2half` / `__high2half` / `__lows2half2` / `__highs2half2` / `__halves2half2` / `__lowhigh2highlow` | `half2` 的分量抽取 / 拼装 / 交换 |
| `__float2bfloat16` / `__bfloat162float`、`__hadd/__hmul/__hfma/__hmin/__hmax`（bf16 重载） | bf16 对应物（`cuda_bf16.h`） |

> 说明：`half2` / `bfloat162` 是**打包**类型，一条指令算两路；完整清单见 `cuda_fp16.h` / `cuda_bf16.h`。
> fp8（`__nv_fp8_*`）也在本机头文件里（域 08）。

## K. 异步与流水

| 原语 | 一句话语义 | 备注 |
|---|---|---|
| `__pipeline_memcpy_async(dst, src, size)` | 发起一次**异步**拷贝（数据未必就位） | 需 `commit` + `wait` 才可见 |
| `__pipeline_commit()` | 把已发起的拷贝**提交**为一组 | |
| `__pipeline_wait_prior(n)` | 等到**只剩 n 组**未完成 | 流水线常用 `wait_prior(0)` |
| `__pipeline_arrive_on(bar)` | 拷贝完成后在指定 barrier 上到达 | 配合 mbarrier |
| `cuda::memcpy_async` / `cuda::pipeline` / `cuda::pipeline_shared_state` | libcu++ 异步拷贝与流水封装 | 推荐写法 |
| `cp.async`（PTX） | sm_80+ 的异步拷贝指令 | `__pipeline_*` 的底层 |
| `cp.async.bulk` / TMA | sm_90 的批量异步搬运 | 🚫 本机无 |

> ⚠ **易错**：异步拷贝**发起 ≠ 完成**。数据没 `commit`+`wait` 之前不可读；靠 `__syncthreads` 也**不**保证
> 拷贝落地——必须等 `wait`/barrier（域 09）。

## L. 宿主侧运行时 / 启动

| 原语 | 一句话语义 | 备注 |
|---|---|---|
| `kernel<<<grid, block, smem, stream>>>(…)` | 异步**启动**一个 kernel | 启动即返回；顺序不保证 |
| `cudaDeviceSynchronize()` / `cudaStreamSynchronize(s)` | 等到**设备 / 某 stream** 上工作完成 | |
| `cudaStreamWaitEvent(s, e)` | 让 stream `s` 等事件 `e` | 建跨 stream 依赖 |
| `cudaEventRecord/create/synchronize/query` | 打点 / 建事件 / 等 / 查 | 计时与依赖 |
| `cudaMemcpyAsync` | 异步拷贝（可重叠） | 需 stream 语义 |
| `cudaMallocAsync` / `cudaFreeAsync` | 异步分配 / 释放 | stream-ordered |
| `cudaGraph*` | 捕获 + 重放执行图 | 降启动开销 |

> ⚠ **易错**：kernel 是**异步**的——很多错误（越界、非法访问）**在之后**才冒出来；`<<<>>>` 本身不报错，
> 要查 `cudaGetLastError` / 同步点（域 10）。

## M. 特殊寄存器 / 调试 / 其它

| 原语 | 一句话语义 | 备注 |
|---|---|---|
| `clock()` / `clock64()` | 读 SM 时钟计数 | 计时用；跨 SM 不可比 |
| `%laneid` / `%warpid` / `%smid` / `%clock64` | 同名特殊寄存器 | 需 inline PTX（域 07） |
| `__nanosleep(ns)` | 让本线程**睡**约 `ns` 纳秒 | sm_70+，降低空转功耗 |
| `assert(cond)` | 断言（device 端） | 失败即中止 |
| `__trap()` | 显式中断 | |
| `__brkpt()` | 断点（调试） | |
| `__prof_trigger(i)` | profiler 触发点 | 性能标记 |

## 出处

- **CUDA C++ Programming Guide（12.8 归档版）**：§7.14 Atomic Functions、§7.5 Memory Fence Functions、
  §7.22 Warp Shuffle Functions、§7.23 Warp Vote、§7.24 Warp Match、§7.25 Warp Reduce、
  §7.20 Arithmetic Instructions、§7.21 Half Precision、§7.4 Synchronization、§7.18 Read-Only Data Cache Load。
  <https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html>
- **CUDA Math API Reference**（`__fadd_*` / `__fmaf_*` / `__clz` / `__popc` / fast math / half 等）：
  <https://docs.nvidia.com/cuda/cuda-math-api/index.html>
- **PTX ISA v8.7**：§9.7.9.6 `shfl.sync`、§9.7.13 `vote`/`bar`/`membar`、§9.7.15.5 `atom`、
  §9.7.10.8 `ld`、§9.7.10.11 `st`。<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- **本机头文件**（CUDA 12.8）：`device_atomic_functions.h`、`sm_20/32/35/60_atomic_functions.h`、
  `sm_20/30/32/35/61_intrinsics.h`、`crt/device_functions.h`、`crt/device_double_functions.h`、
  `crt/sm_70/80/90/100_rt.h`、`cuda_pipeline_primitives.h`、`cuda_fp16.h`、`cuda_bf16.h`；
  libcu++ 侧 `atomic` / `<cuda/atomic>`。
