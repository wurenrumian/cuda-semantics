---
title: 域 05 — 原子语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-08
tags: [cuda, semantics, atomics, memory-model, sm_89]
related: [docs/plan/cuda-semantics-topic-map.md, experiments/EXP-0001-simt-divergence, tmp/outputs/D05]
---

# 域 05 — 原子语义

> 目标机：RTX 4060 Laptop（**sm_89 / Ada**）、CUDA 12.8（nvcc V12.8.61）、driver 596.49。
> 实验代码与产物：`tmp/outputs/D05/`。出处见文末。

## 一句话

CUDA 的原子只承诺**「在某个作用域内，一次读-改-写不可分割」**；默认 intrinsic（`atomicAdd` 等）是
**relaxed 顺序 + device 作用域**，它**不**附带任何内存序、不充当屏障、不建立 happens-before——
「原子」和「有序」是两件正交的事。

## 承诺与边界

**承诺（promise）**

- **原子性（atomicity）**：单个 RMW（read-modify-write）在指定 scope 内不可分割，不会有别的
  RMW 插进它的读与写之间。仅此而已。
- **作用域（scope）** 决定「谁」看得到这次原子：`_block`（CTA）/ 默认（device/gpu）/ `_system`（sys）。
  默认（无后缀）= `cuda::thread_scope_device`（PTX `.gpu`），**不是** system。
- **类型/宽度**（本机 header 实测，见 §本机验证）：
  - 整型 32/64 位：`int` / `unsigned` / `unsigned long long`（`device_atomic_functions.h`）。
  - 浮点：`atomicAdd(float)`（sm_20+，`sm_20_atomic_functions.h`）、`atomicAdd(double)`（sm_60+，
    `sm_60_atomic_functions.h`）。
  - **128-bit（向量 `float2`/`float4`）只有 sm_90+**（`sm_90_rt.h`）——本机 sm_89 **编译不过**
    （见反例/验证）。
- **落点**：global 原子在 **L2 的原子单元**解析；shared 原子在 SM 内共享内存解析。所以 shared 原子
  便宜、global 原子贵——但这是**架构相关的性能语义**（域 12），不是正确性承诺。

**边界（boundary）——原子不承诺什么**

- **不承诺内存序**。指南 §7.14 原话：*"The atomic functions described in this section have
  ordering `cuda::memory_order_relaxed` and are only atomic at a particular scope"*。
  默认缺省 qualifier 的 PTX 是 `atom.global.add.u32`（relaxed + `.gpu`，无 acquire/release）；
  要序必须另加 `__threadfence*()` 或 `cuda::atomic_ref` 的 order。
- **不承诺浮点结果可复现**。`atomicAdd(float/double)` 的结果依赖各线程到达顺序；浮点加法不满足结合律，
  同一输入不同调度可得到不同的低位。**别把浮点原子当确定性的累加器。**
- **不承诺返回值语义之外的任何东西**。返回值是**旧值**（`old`），不是新值、不是总数。
- `_system` 原子在 system scope 下的原子性还有**附加条件**（PTX ISA §8.1.1
  *Limitations on atomicity at system scope*）——本机单卡不做实测。
- **默认 intrinsic 没有 order 参数**；想写「语义正确」的原子要用 libcu++ 的 `cuda::atomic` /
  `cuda::atomic_ref`（可指定 scope + order），它会被编译成 `atom.<order>.<scope>` / `ld.acquire` /
  `st.release` / `fence`。

**两个正交维度**（记住这张表就不会混）：

| 维度 | 取值 | 默认 | 表达方式 |
|---|---|---|---|
| 原子性 | 是/否 | — | `atomicAdd` vs `x = x + 1` |
| 作用域 scope | cta / gpu / sys | gpu(device) | `_block` / 无后缀 / `_system` |
| 顺序 order | relaxed / acquire / release / acq_rel / seq_cst | **relaxed** | `__threadfence*()`、`cuda::atomic_ref` |

## 反例

### 反例 1（核心，语义级）：「用了 atomic 就建立了同步/顺序」

这是最贵的误解。经典 flag+payload 生产者-消费者：

```cpp
// producer: 写 payload（普通 store），再 relaxed 地放旗
for (int i=0;i<n;i++) data[i] = i+1;
atomicExch(&ready, 1);          // 默认 = relaxed + device，不排序上面那批 store

// consumer: relaxed 地自旋等旗，再读 payload
while (atomicAdd(&ready,0)==0) {}
int bad=0; for (int i=0;i<n;i++) if (data[i]!=i+1) ++bad;
```

默认 `atomicExch` 是 relaxed：它**不**保证 producer 的 `data[]` 对 consumer 可见，也**不**保证
consumer 的 `data[]` 读发生在看到 `ready` 之后。这是一个**数据竞争（UB）**，正确写法要
producer `__threadfence()`（release）＋ consumer `__threadfence()`（acquire），
或用 `cuda::atomic_ref` 的 release store / acquire load。

**本机证据（同一份二进制，指令级可复现）**：

- default `atomicAdd`/`atomicExch` 的 PTX 是 `atom.global.add.u32` / `atom.global.exch.b32`，
  **无任何 order qualifier**；SASS 是 `ATOMG.E.ADD.STRONG.GPU`，**前后没有 MEMBAR**。
- 加 `__threadfence()` 后，SASS 里才出现 `MEMBAR.SC.GPU ; ERRBAR ; CCTL.IVALL`。在同一个
  `pub_producer` 里，`useFence=0` 直接跳到 `ATOMG.E.EXCH`，`useFence=1` 才执行 MEMBAR——
  同一份编译产物，差别就是那条 MEMBAR。
- **功能测试反而测不出来**：本机 publish 实验跑了 **2400 轮 0 次读错**（`badRounds=0`）。
  这正是它的危险之处：**不报错的 UB**。硬件通常按序把 store 排到 L2，所以「跑对了」不等于「对」。
  判定靠规范 + SASS/PTX，不靠测试。

### 反例 2（确定性、当场翻车）：把 `float` 位模式当 `int` 用 `atomicMax`

流行技巧是「浮点没有 `atomicMax`，那就把 `float` 重新解释成 `int` 再 `atomicMax`」。它对
**非负** 浮点碰巧成立（IEEE-754 正数的位模式随数值单调），但**两个负数就错**：

```cpp
atomicMax((int*)out, __float_as_int(v));   // 想当然
```

- `-5.0f = 0xC0A00000`，`-3.0f = 0xC0400000`。按**有符号 int** 比较，`0xC0A00000 < 0xC0400000`，
  于是「较大者」被判成 `-5.0f`。
- 本机实测：`max(-5.0, -3.0)` → naive 给 **-5.0（WRONG）**；正确写法（`atomicCAS` 循环 +
  `fmaxf`）给 **-3.0**。
- 更阴的是：正数对、混合符号也对，只有「两个负数」错——只测用例很容易漏。

（`atomicMax` 的文档签名只有整型重载，PTX 也没有 `atom.max.f32`；「浮点原子 max」本来就得自己用
`atomicCAS` 构造。`atomicCAS` 是**通用原子**，可自建任意 RMW，代价是在冲突下自旋重试。）

### 反例 0（底座）：「`x = x + 1` 看起来像一条原子写」

源码一行，汇编是 load + add + store 三步。本机 256 线程 ×2000 次：预期 512000，非原子版只剩
**2000**（丢 510000）。原子性不是「编译器会自动帮我做对」。

## 本机验证（命令 + 观察）

所有产物在 `tmp/outputs/D05/`。源码：`atomics_lab.cu`（功能）、`atomics_sass.cu`（指令）、
`atomics_b128_sm90.cu`（边界）、`sanitizer_smoke.cu`（工具自检）、`run.ps1`（一键跑）。

### 复现步骤

```powershell
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$out    = 'E:\Project\cuda-semantics\tmp\outputs\D05'

# 功能实验
cmd /c "call `"$vcvars`" && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$out\atomics_lab.exe`" `"$out\atomics_lab.cu`""
& "$out\atomics_lab.exe" atomicity
& "$out\atomics_lab.exe" floatmax
& "$out\atomics_lab.exe" publish 0 2000     # 无反例：0 bad；加 fence 也是 0

# 指令级：PTX + SASS
cmd /c "call `"$vcvars`" && `"$nvcc`" -arch=sm_89 -ptx   -o `"$out\atomics_sass.ptx`"   `"$out\atomics_sass.cu`""
cmd /c "call `"$vcvars`" && `"$nvcc`" -arch=sm_89 -cubin -o `"$out\atomics_sass.cubin`" `"$out\atomics_sass.cu`""
& $cuobj -sass "$out\atomics_sass.cubin" > "$out\atomics_sass.sass"
& $cuobj -sass "$out\atomics_lab.exe"    > "$out\atomics_lab.sass"   # 真实内核（publish）的 SASS

# 边界：128-bit 原子在 sm_89 应当编译失败，sm_90 通过
cmd /c "call `"$vcvars`" && `"$nvcc`" -arch=sm_89 -cubin -o `"$out\atomics_b128_sm89.cubin`" `"$out\atomics_b128_sm90.cu`""
cmd /c "call `"$vcvars`" && `"$nvcc`" -arch=sm_90 -cubin -o `"$out\atomics_b128_sm90.cubin`" `"$out\atomics_b128_sm90.cu`""
```

### 观察 1 — 设备/能力

`run_props.txt`：`RTX 4060 Laptop GPU`，compute capability **8.9**，warpSize 32，
SM 数 24，L2 **32 MiB**。

### 观察 2 — PTX：默认原子没有 order，scope 靠后缀

`atomics_sass.ptx`（节选，均为原文）：

| 写法 | 生成的 PTX |
|---|---|
| `atomicAdd(&g,1)` | `atom.global.add.u32`（无 qualifier ⇒ relaxed + `.gpu`） |
| `atomicExch(&g,1)` | `atom.global.exch.b32` |
| `atomicCAS(&g,a,b)` | `atom.global.cas.b32` |
| `atomicAdd_block` | `atom.global.cta.add.u32` |
| `atomicAdd_system` | `atom.global.sys.add.u32` |
| 共享内存 `atomicAdd` | `atom.shared.add.u32` |
| `atomicAdd(float)` / `(double)` | `atom.global.add.f32` / `atom.global.add.f64` |
| `__threadfence()` | `membar.gl` |
| `__threadfence_block()` | `membar.cta` |
| `__threadfence_system()` | `membar.sys` |
| `cuda::atomic_ref` relaxed `fetch_add` | `atom.add.relaxed.gpu.s32` |
| 同 `seq_cst` | `fence.gpu.sc;` + `atom.add.acquire.gpu.s32` |
| release `store` | `st.release.gpu.b32` |
| acquire `load` | `ld.acquire.gpu.b32` |

### 观察 3 — SASS：fence 与原子在指令层是分开的

`atomics_sass.sass` / `atomics_lab.sass`：

| 写法 | 生成的 SASS |
|---|---|
| default `atomicAdd`/`exch`/`cas` | `ATOMG.E.ADD/EXCH/CAS.STRONG.GPU`，**无 MEMBAR** |
| 返回值未用 | 降为 `RED.E.ADD.STRONG.GPU`（`atom` → `red`） |
| 共享内存 `atomicAdd` | `ATOMS.ADD` |
| `atomicAdd(float)` | `ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU` |
| `atomicAdd(double)` | `ATOMG.E.ADD.F64.RN.STRONG.GPU` |
| `atomicAdd_system` / `_block` | `...STRONG.SYS` / `...STRONG.SM` |
| `__threadfence()` | `MEMBAR.SC.GPU ; ERRBAR ; CCTL.IVALL` |
| `__threadfence_block()` / `_system()` | `MEMBAR.SC.CTA` / `MEMBAR.SC.SYS` |
| acquire load / release store | `LD.E.STRONG.GPU` + `CCTL.IVALL` / `MEMBAR.ALL.GPU` + `ST.E.STRONG.GPU` |

真实内核 `pub_producer` 的 SASS 里，`useFence` 运行时判断：`@!P0 BRA` 跳过
`MEMBAR.SC.GPU; ERRBAR; CCTL.IVALL`，否则执行后才 `ATOMG.E.EXCH`。**「原子」和「栅栏」在指令流里
是两条分开的指令**，这就是「原子 ≠ 同步」的机器级证据。

### 观察 4 — 功能数值

| 实验 | 期望 | 实测（`run_*.txt`） |
|---|---|---|
| 原子性 `atomicity` | 512000 | 非原子 2000（丢 510000）；`atomicAdd` 512000 |
| 共享竞争 `race-naive` | 8192 | **256** |
| 共享竞争 `race-atomic` | 8192 | 8192 |
| 浮点 max `floatmax` | -3.0 | naive **-5.0（WRONG）**；CAS+fmaxf -3.0 |
| 发布 `publish 0 2000` | 0 | **0**.badRounds（无反例——静默 UB） |
| 边界 `b128_sm89` | 编译失败 | `error: no instance of overloaded function "atomicAdd" matches … (float4 *, float4)` |
| 边界 `b128_sm90` | 通过 | 生成 `atomics_b128_sm90.cubin` |

### 观察 5 — compute-sanitizer：本机**不可用**（环境限制）

`E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe`（2025.1.0）反复报：

```
========= COMPUTE-SANITIZER
========= Error: process didn't terminate successfully
========= Error: Target application terminated before first instrumented API call
```

**连一个 trivial 的 `cudaMalloc`+kernel 程序（`sanitizer_smoke.cu`）也一样**，`racecheck` /
`synccheck` / `memcheck` 全部如此。已排除：路径、`--injection-path`、`--target-processes all`、
`CUDA_INJECTION64_PATH`、复制到 CS 目录、改用 `cmd`/`Start-Process`。判断是**注入失败**：CS 需要
`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface=1`（`EnableDebuggerInterface.bat`），
而当前会话**非管理员**，`reg add` 返回 *Access is denied*。
→ 本域**没有**拿到 racecheck 报告；竞争/可见性的证据改用 SASS/PTX（观察 2/3）。

> 顺带纠正记忆：`EXP-0001` 的 README 曾记「racecheck 通过」，但 `tmp/outputs/EXP-0001/` 并不存在，
> `STATE.md` 也写「未跑出结果」——本机 racecheck 目前跑不起来，那条结论无产物支撑。

### 产物清单

| 文件 | 内容 |
|---|---|
| `run.ps1` | 一键构建 + 全模式 + PTX/SASS + 边界编译 |
| `atomics_lab.cu` / `atomics_sass.cu` / `atomics_b128_sm90.cu` / `sanitizer_smoke.cu` | 源码 |
| `run_props.txt` / `run_atomicity.txt` / `run_floatmax.txt` / `run_publish_0_*.txt` / `run_race-*.txt` | 功能输出 |
| `atomics_sass.ptx` / `atomics_sass.sass` / `atomics_lab.sass` | 指令级证据 |
| `b128_sm89_build.log` / `b128_sm90_build.log` | 边界编译日志 |
| `racecheck_*.txt` | sanitizer 失败记录 |
| `refs/SOURCES.md` | 出处 URL |

## 出处

- **CUDA C++ Programming Guide（12.8 归档版）§7.14 Atomic Functions**（含 §7.14.1.1 `atomicAdd`、
  §7.14.1.3 `atomicExch`、§7.14.1.5 `atomicMax`、§7.14.1.8 `atomicCAS`）——
  「have ordering `cuda::memory_order_relaxed` and are only atomic at a particular scope」，
  以及 `_system` / 无后缀 / `_block` 三档作用域的说明。
  <https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html>
- 同上 **§7.5 Memory Fence Functions**（`__threadfence*` 的作用与消息传递用法）。
- 同上 **§9.2.2 Memory Model**、**§9.2.1.2 Scope of CUDA Primitives**。
- **CUDA Programming Guide（重构版）§5.7 CUDA C++ Memory Model**：5.7.1 Thread Scopes、
  5.7.2 Synchronization primitives、5.7.3 Atomicity、5.7.4 Data Races、**5.7.5 Example: Message Passing**。
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html>
- **PTX ISA v8.7**：§8 Memory Consistency Model（§8.1 Scope and applicability、§8.1.1 Limitations on
  atomicity at system scope、§8.5 Scope、§8.7 Morally strong operations、§8.9 Ordering of memory
  operations、§8.10.3 Atomicity）；**§9.7.15.4 membar/fence**；**§9.7.15.5 atom**；**§9.7.15.6 red**；
  §9.7.10.8 `ld` / §9.7.10.11 `st`（acquire/release）。
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- 本机 header：`include/crt/sm_20_atomic_functions.h`（`atomicAdd(float)`）、
  `sm_60_atomic_functions.h`（`atomicAdd(double)`）、`device_atomic_functions.h`（64 位整型）、
  `sm_90_rt.h`（`atomicAdd(float2/float4)`，128-bit，sm_90+）。

## 未决

- **compute-sanitizer 本机不可用**：需管理员开启 `GPUDebugger\EnableInterface`，或用其它注入前置条件；
  待补 `racecheck` 对本域「可见性/竞争」的独立确认。这是一条**需要复现的环境阻塞**。
- **反例 1 未在本机复现**（0/2400）。要把它变成「跑得出错」的 litmus，需要跨 SM 的 store-buffer /
  message-passing 变体（域 4 的核心工具），本机尚未搭。当前判定依赖规范 + SASS/PTX。
- **`ATOM…STRONG.GPU` 里 `STRONG` 与 relaxed/acquire 的精确对应**未厘清：本机 relaxed 与 acquire
  的 ATOM 都显示 `.STRONG.GPU`，区别只在 `MEMBAR`/`CCTL.IVALL`。需要对照 `ld.relaxed` vs
  `ld.acquire` 的 SASS 才能定论（可能属域 4/7）。
- **`atomicAdd(float)` SASS 出现 `.FTZ`**（默认未开 `-ftz`）：是硬件对 float 原子固定 FTZ，还是
  编译选择？待用 `-ftz=true/false` 对照确认（跨域 8）。
- **128-bit 原子（`.b128`）在 sm_90 的行为**未实测——本机无 sm_90，只证到「sm_89 编译失败」。
- **浮点原子求和的可复现性**只在概念层（非结合），未做「同输入不同分块得不同低位」的最小实验。
- `_system` 原子的 system-scope 原子性附加条件（PTX §8.1.1）本机单卡不做实测。
