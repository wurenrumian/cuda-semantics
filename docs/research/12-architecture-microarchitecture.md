---
title: 域 12 — 架构 / 微架构相关的语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-12
tags: [cuda, semantics, architecture, microarchitecture, sm_89, ada, sass, occupancy, shared-memory, coalescing, sector]
related:
  - docs/plan/cuda-semantics-topic-map.md
  - experiments/EXP-0012-architecture-microarchitecture
  - tmp/outputs/D12/2026-10-08/
  - docs/research/01-simt-execution-model.md
  - docs/research/02-thread-hierarchy.md
  - docs/research/03-memory-spaces.md
  - docs/research/09-async-pipelines.md
---

# 12 — 架构 / 微架构相关的语义（ISA 承诺 vs 架构决定）

> 域内定位：`docs/plan/cuda-semantics-topic-map.md` §12。
> 目标机：**sm_89（Ada, RTX 4060 Laptop，24 SM）**，**CUDA 12.8**（`nvcc V12.8.61`），driver 596.49。
> 源码与产物：`experiments/EXP-0012-architecture-microarchitecture/`、`tmp/outputs/D12/2026-10-08/`（`arch_probe.out`、`sass_sm_{75,80,89}.txt`、`sass_cpasync_sm89.txt`、`build-*.log`、`coalesce-*.txt`、`ncu-*.txt`）。
> 一键复现：`pwsh -File experiments\EXP-0012-architecture-microarchitecture\scripts\run.ps1`。

## 一句话

CUDA 的语义分两层：一层是 **ISA（PTX / CUDA C++ 抽象机）承诺的**，跨架构不变；另一层由**具体架构实现决定**，换一块 GPU 就可能改。本域的坑几乎都来自"把后一层当承诺用"：warp 大小（32）是 ISA 常量，而 SM 数量、每 SM 的资源上限、block 落到哪个 SM、SASS 指令序列、访存代价、原子吞吐，全都只是**可查询 / 可观测的架构事实**。

## 四层分家：每层给自己的"字据"不同

| 层 | 内容 | 来源 | 怎么验证 | 能不能当依据写进推理 |
|---|---|---|---|---|
| ① ISA / 编程模型保证 | warp=32、内存模型、原子性、同步语义 | 规范（域 01–11 各篇已引） | 规范原文 + 各域实测 | **可以**（跨架构） |
| ② 可查询的架构属性 | SM 数、每 SM 上限、L2、`concurrentKernels` | `cudaDeviceProp` / `cudaFuncAttributes` | 运行时查询 | 可以**描述当前设备**，不能写成"永远如此" |
| ③ 微架构实现细节 | SASS 指令形态、指令选择、发射方式 | -cubin + cuobjdump（本机 D07/这里） | 同源码多 arch 对照 | 只能说"这份编译产物这样"，**不当语义条约** |
| ④ 性能测量 | 时延、吞吐、占用率曲线 | 计时/计数器（本域的 coalesce 探针） | 可复现口径 + 多次重复 | 只结论"**本机当前状态**"，换设备重测 |

用例的关系：③ 只在"证实 ② 的解释"或"证伪直觉"时有用（反例一）；④ 的数字离开 ②③ 的
口径（访问模式/工作集/缓存状态）没有意义（反例三的口径提醒）。**查得到 ≠ 可依赖**：
正确性永远回 ①。

## 承诺与边界

### ISA 层：可以依赖的

- **warp = 32 线程**：`warpSize == 32`，跨架构固定，是 ISA 常量。
- **PTX 前向兼容，cubin 架构特化**：为 `compute_XY` 生成的 PTX 可由更新驱动 JIT 到未来架构；而 **cubin** 只在"同一 major、目标 minor ≥ 编译时 minor"的范围内二进制兼容，跨 major（如 sm_75 → sm_89）不能直接加载。
- **`.target` / `-arch` 门控指令可用性**：某条指令在某架构上"存不存在"由编译目标决定（例：`cp.async` 要求 sm_80+；TMA / tensormap 要求 sm_90a）。这是**硬边界**：低于门槛则编译期报错，而不是运行期行为差异。
- **抽象机语义不随架构变**：域 01–11 的执行模型 / 内存模型 / 原子性 / 同步承诺，在任何 `sm_XY` 上一样成立。

### 架构层：能查询、但**不是**承诺

`cudaDeviceProp` / `cudaFuncGetAttributes` 给出的只是"**这块** GPU 的事实"（本机 `arch_probe.out`）：

| 量 | 本机（sm_89）观察值 | 性质 |
|---|---|---|
| `warpSize` | 32 | ISA 常量（跨架构） |
| `multiProcessorCount` | 24 | 架构事实 |
| `maxThreadsPerMultiProcessor` | **1536**（≠ 常见资料里的 2048） | 架构事实 |
| `maxBlocksPerMultiProcessor` | 24 | 架构事实 |
| `regsPerMultiprocessor` / `regsPerBlock` | 65536 / 65536 | 架构事实 |
| `sharedMemPerMultiprocessor` | 102400（100 KB） | 架构事实 |
| `sharedMemPerBlock`（静态上限） | 49152（48 KB） | 语义 / 工具链边界 |
| `sharedMemPerBlockOptin`（动态可 opt-in 上限） | 101376（99 KB） | 语义 / 工具链边界 |
| `reservedSharedMemPerBlock` | 1024 | 架构事实 |
| `l2CacheSize` | 33554432（32 MiB） | 架构事实 |

### 三道门：从 compute capability 到"这个特性我能用吗"

把"能不能用某特性"的疑问拆成**串行的三道门**，每道门验收手段不同——跳过一扇就错答：

| 门 | 问题 | 验收手段 | 例子 |
|---|---|---|---|
| ① **编译能过吗** | 目标 `-arch` 的工具链**接受**这个写法吗 | `nvcc -arch=sm_XX` 试编（backend ptxas 门控报错） | `cp.async` 在 sm_89 报错？反了——sm_89 过、sm_75 报（观察 C） |
| ② **这块设备支持吗** | 运行时机器**有**这个能力吗 | `cudaDeviceProp`（major/minor、各 attr）或 `cudaDeviceGetAttribute` | TMA：sm_90 门控 → 本机 sm_89 **两道门都过不去** |
| ③ **运行期真用得动吗** | opt-in / context / 权限齐了吗 | `cudaFuncSetAttribute` 返回码、运行期实测 | 99KB 动态 shared：过了①②还需 opt-in（观察 B），缺它 `invalid argument` |

三道门各自失败的信息也不同：① ptxas 报错（编译期）；② 查询属性回 0 或假；③ API 返回
`invalid argument`/运行时错。常见的错答是拿 ① 的"编译过了"当 ② 的"一定支持"（fatbin 里
塞了 PTX 时尤其迷惑——见下 JIT 小节），或拿 ② 的"属性查到"当 ③ 的"不用 opt-in 就能用"。

**门控 vs JIT fallback（别混为一对邻居）**：-arch/-gencode/-gpu-code 决定 fatbin 里装什么
（SASS for sm_XX + PTX for compute_XX 各选）；新架构跑旧 fatbin 时走 **PTX JIT**。JIT 的
语义仍靠 PTX 层兑现，但**性能/指令选择是 JIT 现场定的**（分类里的 ③层换个生成者）——
"PTX 兜底能跑"不等于"性能按源 SASS 预期"。域 07 的四层链路图给这层关系做底图。

**边界清单（想当然就会错的）：**

- **`maxThreadsPerMultiProcessor` 不是 2048**：Ada（8.9）是 1536，occupancy 上限天然低于 Ampere 的 2048。
- **"硬件有 100 KB shared/SM" ≠ "一个 block 能用 100 KB 静态 shared"**：静态 `__shared__` > 48 KB 编译期就被 ptxas 拒绝；要到 101376 B 必须写**动态** shared 并 `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)` opt-in。
- **`-arch=sm_89a` / `-arch=sm_89f` 不是随处可用的后缀**：`a` = architecture-specific（只对那一个 compute capability 可用），`f` = family-specific（同一 GPU family 内可用）；后缀和架构的合法组合随 toolkit / 架构而变。本机 nvcc 12.8 两者都拒绝，而且 12.8 连 `sm_100f` 也不认——`f` 后缀在这个版本里还没落地。
- **SASS 不是语义契约**：同一份 CUDA C++ 在不同 `-arch` 下落到**不同指令序列**；SASS 是各架构实现，不是 ISA 承诺。
- **访存代价、原子吞吐、bank 冲突、占用率曲线是"性能语义"**：能测、会随架构变，**不能**写成正确性假设。

## 反例

### 反例一（主）：以为"同算法 ⇒ 同机器码"——同一 kernel 跨 sm_75/80/89 的 SASS 不同

"我就是调了个 `atomicAdd` 罢了，SASS 能差到哪去" —— 实际上连指令都换了。同一个 `atom_probe`（共享内存 `atomicAdd`）在三个目标下（`sass_sm_{75,80,89}.txt`），按指令类别对齐摘录如下（省略了重复的屏障行，不是逐行原样）：

```
sm_75 :  @!P0 STS [RZ], RZ ;   ATOMS.ADD RZ, [RZ], R0 ;        BAR.SYNC 0x0 ;                LDS.U R0, [RZ] ;   STG.E.SYS [UR4], R0 ;
sm_80 :  @!P0 STS [RZ], RZ ;   ATOMS.POPC.INC.32 RZ, [URZ] ;   BAR.SYNC.DEFER_BLOCKING 0x0 ;  LDS R5, [RZ] ;     STG.E [R2.64], R5 ;
sm_89 :  （与 sm_80 相同）
```

- 共享原子：**`ATOMS.ADD`（Turing）→ `ATOMS.POPC.INC.32`（Ampere+）**——Ampere 多了条"按活跃掩码 popcount 递增"的原子指令，寻址也从普通寄存器换成 uniform 寄存器 `URZ`。
- 屏障：`BAR.SYNC` → `BAR.SYNC.DEFER_BLOCKING`。
- 只读访存 / 写回的作用域标注：`LDG.E.CONSTANT.SYS`、`STG.E.SYS`（sm_75）→ `LDG.E.CONSTANT`、`STG.E`（sm_80/89）。

`__ldg` 的 `sass_probe` 还暴露了常量物化的差异：sm_80 用 `HFMA2.MMA R7,-RZ,RZ,0,2.38e-07` 造出 4，sm_89 直接 `MOV R7, 0x4`；sm_75 则是 `MOV R1, c[0x0][0x28]`，sm_80/89 用 `IMAD.MOV.U32`。**结论：把具体 SASS 序列当作跨架构不变量是错的；保证只在 PTX / CUDA C++ 那一层。**

> **对照口径**（这类跨架构比较怎么才算数）：三份 SASS 出自**同一源码、同一编译开关**
> （`-O3 -cubin`，只换 `-arch`），对照按"指令类别对齐"摘录而非逐行——否则 diff 里全是
> 伪差异。"解释了差异"与"升级成语义要求"是两步：**本表只做前者**。任何"某代必须有
> XXX 指令"的写法都是把 ③ 层观察写成 ① 层承诺，按四层分家表是错级的。

### 反例二："硬件有 100 KB shared，我放 104 KB 静态数组就行"——编译期直接被拒

"`sharedMemPerMultiprocessor = 102400`，那我 `__shared__ float big[26000]`（104000 B）当然可以" —— ptxas 直接编译失败（`build-static_over.log`）：

```
ptxas error : Entry function '_Z15too_much_staticv' uses too much shared data (0x19640 bytes, 0xc000 max)
```

`0x19640 = 104000`，`0xc000 = 49152 = 48 KB`。**"能不能放"是分层的：静态 48 KB（编译期）< opt-in 动态 101376（运行期）≤ 硬件 100 KB/SM。** 把硬件容量当成"一块 block 能用多少 shared"就会错。

同一组数字，从 occupancy 视角再读一遍（每级限制**各管一摊**， 取的是最小）：

```
硬件/SM 上限        102400  ← 池子容量的家用上限
   ▲
block 可请求上限    101376  ← opt-in 后允许"一块 block"要这么多
   ▲
静态编译上限         49152  ← 静态 __shared__ 直接锚定这一层
```

- **静态/动态在同一池子**：两者都从"每 SM 的 shared 池"里扣（02 域 dyn_smem 实验：静态
  16KB 与动态 16KB 都给 5 block/SM，同池实锤）。
- **opt-in 只是"允许"**：`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 解除的是
  "单 block 请求量"的门（49152→101376），**不改变**池子总量——SM 同一时刻塞几个 block
  由此还要除以（动态量 × block数 ≤ 池子；`__launch_bounds__` 的 minBlocks 是**下界不封顶**，
  见 02 域"launch_bounds 实验"）。
- **`reservedSharedMemPerBlock` = 1024 B**：每块还得留 1KB 系统 reserved——报告"可用"时它也
  算在你的申请单里（本机 arch_probe.out 的观察值之一）。

### 反例三（性能语义，非正确性）：以为"读的元素数一样 ⇒ 代价一样"

同一个探针、**读的元素数完全相同**，只改每个 lane 的地址模式（`coalesce_probe.cu`）：

```
coalesced: 0.0182 / 0.0136 ms/iter
strided  : 0.1790 / 0.1790 ms/iter        ≈ 10–13×
```

原因在 32B sector 粒度：连续 warp 触碰 4×32B，跨步 warp 每个 lane 各自一个 sector。**这条只能当"性能语义"用**——由架构 / 访存层级决定、随架构变；把它当成"跨步一定慢 10 倍"写进正确性推理，就是另一类错。

（**口径提醒**：两种模式读的元素数一样，但**工作集不一样**——连续只碰前 4 MB（能常驻 32 MB 的 L2），跨步要碰满 128 MB（基本走显存）。所以这个 10–13× 里还混着"L2 命中 vs 走 DRAM"的成分，不是纯粹的 sector 计数对比；要分离二者得看 `ncu` 的 sector 计数器，而本机拿不到。）

> **这条性能实验的完整口径**（引用数字前先核对五要素，按四层分家的 ④ 层要求）：
> **访问模式** = 连续/跨步（`coalesce_probe.cu` 内定）；**缓存状态** = L2 常驻与否已混入数字
> （下面的工作集差异）；**工作集** = 128 MB（跨步）/ 有效约 4 MB（连续，见口径提醒）；
> **计时方法** = 墙钟 ms/iter（无计数器）；**重复次数/取值方式** = 以 `scripts/run.ps1`
> 与源码为准。缺任何一项，这数字就**不能**被引用去支撑别的结论——尤其别外推到其他 GPU
> 或其他访存/工作集组合。

（还有一条同属"环境 / 权限语义"的坑：`ncu` 读合并访存计数器在本机被拒，见下。）

## 本机验证（命令 + 观察）

### 环境与编译

```bat
call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -arch=sm_89 -o arch_probe.exe scripts\arch_probe.cu
arch_probe.exe > arch_probe.out
```

一键复现：`pwsh -File experiments\EXP-0012-architecture-microarchitecture\scripts\run.ps1`。

### 观察 A：可查询的架构事实（`arch_probe.out`）

`maxThreadsPerMultiProcessor = 1536`（不是 2048），occupancy 随 block 大小变化：

```
block= 128 -> 12 block/SM = 1536 threads/SM (100.0%)
block= 256 ->  6 block/SM = 1536 threads/SM (100.0%)
block= 640 ->  2 block/SM = 1280 threads/SM ( 83.3%)
block=1024 ->  1 block/SM = 1024 threads/SM ( 66.7%)
```

`%smid`：launched 192 blocks × 128 threads on a 24-SM device → `distinct SMs observed: 24`，`smid range observed: [0, 23]`。block 会被摊到各 SM，但**"哪个 block 落到哪个 SM"不是承诺**。

### 观察 B：静态 shared 上限 vs 动态 opt-in（`build-static_over.log`、`arch_probe.out`）

```
# 静态 104000 B：编译期失败
ptxas error : ... uses too much shared data (0x19640 bytes, 0xc000 max)

# 动态 101376 B：
dynamic smem=101376 without opt-in : invalid argument
cudaFuncSetAttribute(MaxDynamicSharedMemorySize=101376) : no error
dynamic smem=101376 WITH opt-in     : no error
```

### 观察 C：指令可用性由编译目标门控（`build-cpasync_*.log`、`sass_cpasync_sm89.txt`）

同一份 `cpasync_gate.cu`（inline asm `cp.async`）：

```
# -arch=sm_75
ptxas error : Feature 'cp.async' requires .target sm_80 or higher
ptxas error : Feature 'cp.async.commit_group' requires .target sm_80 or higher
ptxas error : Feature 'cp.async.wait_group' requires .target sm_80 or higher

# -arch=sm_89 → 通过，落到 SASS：
LDGSTS.E.128 [RZ], [R2.64] ;
LDGDEPBAR ;
DEPBAR.LE SB0, 0x0 ;
```

### 观察 D：跨 sm_75/80/89 的 SASS 对比

```
nvcc -O3 -cubin -arch=sm_75 -o sass_probe_sm_75.cubin scripts\sass_probe.cu
nvcc -O3 -cubin -arch=sm_80 -o sass_probe_sm_80.cubin scripts\sass_probe.cu
nvcc -O3 -cubin -arch=sm_89 -o sass_probe_sm_89.cubin scripts\sass_probe.cu
cuobjdump -sass sass_probe_sm_75.cubin > sass_sm_75.txt     # 同理 sm_80 / sm_89
```

观察见"反例一"：`ATOMS.ADD` → `ATOMS.POPC.INC.32`、`BAR.SYNC` → `BAR.SYNC.DEFER_BLOCKING`、`.SYS` 标注的有无、常量物化方式不同。

### 观察 E：架构后缀的接受度（`build-suffix_*.log`）

```
-arch=sm_89a : nvcc fatal : Unsupported gpu architecture 'sm_89a'
-arch=sm_89f : nvcc fatal : Unsupported gpu architecture 'sm_89f'
-arch=sm_90a : OK（生成 sm_90a cubin）
```

同一次实测里 `sm_100f` 也被拒：`nvcc fatal : Unsupported gpu architecture 'sm_100f'`。可见 12.8 认不出 `f` 族后缀，并不是只对 Ada 拒绝。

### 观察 F：合并访存计时 + ncu 权限（`coalesce-*.txt`、`ncu-*.txt`）

```
coalesced: 0.0182 / 0.0136 ms/iter
strided  : 0.1790 / 0.1790 ms/iter

ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,... coalesce_probe.exe coalesced
  ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
            GPU Performance Counters on the target device 0.
```

**注意：** 合并访存的"sector 级"证据本机**没拿到**（计数器权限被拒），只有毫秒级的墙钟差异；32B sector 是文档层语义，本次只是与之**定性一致**的观察，不是直接证明。

## 出处

**CUDA C++ Programming Guide（12.8）**

- **Compute Capabilities → Features and Technical Specifications**：每架构的每 SM 线程数 / 每 SM 与每 block 的 shared 上限 / 32 个 shared bank / 每 SM 寄存器数；同页 **Feature Set Compiler Targets** 解释 `a` 后缀 = architecture-specific（仅该 compute capability 可用）、`f` 后缀 = family-specific（同一 GPU family 内可用），示例 `compute_100f` / `compute_100a`。
  `docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html`
- **Hardware Implementation → SIMT Architecture / Hardware Multithreading**：warp = 32、每 SM 的 warp 槽位、用多 warp 隐藏延迟。
- **Programming Model → Thread Hierarchy / Memory Hierarchy**：thread/block/grid 抽象与内存空间的抽象承诺。
- **Writing CUDA Kernels → Coalesced Global Memory Access**：warp 内访问映射到 **32-byte memory segments**（合并的定义与目标）。
  `docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html`
- **Compilation with NVCC**：`-arch` / `-gencode` / `-gpu-code`（虚拟架构 `compute_XY` vs 真实架构 `sm_XY`），以及 **Binary Compatibility / PTX Compatibility**（cubin 架构特化、PTX 前向兼容）。
  `docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html`
- **Special Topics → Asynchronous Data Copies**：`cp.async` / `cuda::pipeline`（sm_80+）；TMA / tensormap 需 `sm_90a`。
  `docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html`
- **CUDA Runtime API → Device Management**：`cudaDeviceProp`、`cudaFuncAttributes`、`cudaOccupancyMaxActiveBlocksPerMultiprocessor`、`cudaFuncSetAttribute`。

**PTX ISA**

- **Data Movement and Conversion Instructions → Asynchronous copy → Non-bulk copy**：`cp.async` / `cp.async.commit_group` / `cp.async.wait_group`，及"requires `sm_80` or higher"的门控。
- **Data Movement and Conversion Instructions → `ld` / `st`**：cache operator（`.ca/.cg/.cs/.lu/.cv/.wt`）、`.relaxed/.acquire/.release` 修饰；SASS 里的 `.SYS` 等作用域标注对应这一层。
- **Parallel Synchronization and Communication Instructions → `atom` / `red`**：`.cta/.gpu/.sys` 作用域；global 原子在 L2 的原子单元解析。
- **Memory Consistency Model（§8）**：scope 与 order 的语义来源。
- **Special Registers**：`%smid`、`%warpid` 等只读特殊寄存器。
  `docs.nvidia.com/cuda/parallel-thread-execution/`

## 未决

- **ncu 计数器被权限阻断（ERR_NVGPUCTRPERM）**：拿不到 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 等，合并访存只有墙钟证据。要让观察和文档的 32B sector 语义对上，需管理员开 GPU Performance Counters 权限（NVIDIA 控制面板 / `NVreg_RestrictProfilingToAdminUsers`）。
- **sm_89a / sm_89f 的拒绝是 CUDA-12.8 的实测**，而且 12.8 对任何 `f` 后缀都拒绝（连 `sm_100f` 也是）——所以「出处」里引的 `compute_100f` / `compute_100a` 示例来自比 12.8 更新的在线版指南。至于更新版（如 12.9+）对 Ada 的族目标怎么定义、哪些 (arch, suffix) 组合合法，本机未验证。
- **cubin vs PTX 的加载行为未直接跑**：sm_75 cubin 拿到 sm_89 上应因 "no kernel image" 失败、而 PTX 可 JIT——目前只有 doc 依据，未实测（可作为秒级小验证补上）。
- **shared carveout（L1/shared 可配置划分）未探**：可用 `cudaFuncAttributePreferredSharedMemoryCarveout` 观察；本次只动了 shared 上限。
- **SM 内 sub-partition 结构未直接观测**：scheduler 数、每 scheduler 的 warp 槽位、寄存器文件分区——需要 ncu 的 scheduler 指标（被权限阻断）。
- **架构相关功能单元的吞吐**（SFU / DP / tensor core / `mma` fragment 布局）本次未触及，属域 08 与更细的微架构笔记。
- **原子落点（L2 atomic unit）与 `red` / `atom` 吞吐**未直接测；域 05 的 `publish` 实验只覆盖正确性，不覆盖 L2 解析位置。
