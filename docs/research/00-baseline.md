---
title: 00 环境基线
type: research
status: draft
created: 2026-10-11
updated: 2026-10-11
tags: [cuda, baseline, environment, sm_89, ada, cuda-12.8, toolchain]
related: ["experiments/EXP-0000-baseline", "docs/plan/cuda-semantics-topic-map.md", "INTENT.md"]
---

# 00 · 环境基线

> 目标：把「本机是什么、因此哪些语义适用 / 不适用」钉死，作为 01–12 全部结论的前提。
> 环境：sm_89（Ada, RTX 4060 Laptop）/ CUDA 12.8 / nvcc V12.8.61。
> 本页不是某条语义本身，而是**语义的天花板**：它决定哪些域能实机验证、哪些只能做概念了解。

## 一句话

**本机 = sm_89（Ada）· CUDA 12.8 · 单卡 8 GB。** 三条硬约束里，①② 决定「能跑什么」，③ 决定「工具能不能真正出结果」。
① Volta+ 的**独立线程调度（ITS）**让「warp 锁步」变成 UB；
② **sm_89 < sm_90**：TMA、cluster+DSMEM、128-bit 原子、`cp.async.bulk` 都不可用；
③ 工具链齐全，但本机非管理员，sanitizer 与 ncu 受限。
后续每个域的「本机验证」都在这条基线之内；超出者一律标注「概念了解，不在本机跑」。

## 基线事实（本机实测）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute capability | **8.9（sm_89, Ada Lovelace）** |
| 显存 | 8188 MiB（≈ 8 GB） |
| 驱动 | 596.49 |
| CUDA Toolkit | **12.8**（nvcc V12.8.61），`E:\NVIDIA\CUDA\v12.8` |
| MSVC | VS2022 Community，`E:\Microsoft Visual Studio\2022\Community`（nvcc 须先 `vcvars64.bat`；路径含空格，`call` 时必须加引号） |
| PyTorch | 2.11.0+cu128，`torch.version.cuda == 12.8`，`cuda.is_available() == True`（venv `E:\venvs\pytorch312`） |
| compute-sanitizer | 2025.1.0，`E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe`（**不在 PATH**） |
| Nsight Compute | 2025.1.0（`ncu.bat`，在 PATH） |
| Nsight Systems | 2024.6.2（`C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2`，**不在 PATH**） |
| 静态观察工具 | `ptxas` / `cuobjdump` / `nvdisasm`（均在 PATH） |

> 一键复现：`experiments/EXP-0000-baseline/`（`pwsh -File scripts/run.ps1`）。

## 承诺与边界

### 三条硬约束（贯穿全图）

1. **ITS（Volta+, CC≥7.0）**：本机 sm_89 有独立线程调度 → 依赖「warp 隐式锁步」的旧代码是 **UB**；warp 内同步要用 `__syncwarp()`（域 01/06）。
2. **sm_89 < sm_90**：**无** TMA / `cp.async.bulk` / thread block cluster + distributed shared memory / 128-bit（`.b128`）原子 / `setmaxnreg` 等 Hopper 特性。这些当**概念了解**，不在本机跑（域 06/09/12）。反之 `cp.async`（sm_80+）、fp8 tensor core（sm_89）**有**。
3. **工具链就绪**：`nvcc` / `ptxas` / `cuobjdump` / `nvdisasm` / `compute-sanitizer` / `ncu` / `nsys` 组件齐全 —— 但「齐全」不等于「本机就能出结果」（见下）。

### 不承诺（想当然就会错的地方）

- **工具装了 ≠ 本机能用**：`compute-sanitizer` 与 `ncu` 在 Windows 需要管理员 / 调试接口权限。本机实测 `ncu` 报 `ERR_NVGPUCTRPERM`、`compute-sanitizer` 报 *"Target application terminated before first instrumented API call"*。→ racecheck / synccheck / initcheck 的验证因此受阻（域 04/06/11 的公共阻塞项）。
- **「Ada 支持 Ada 的全部特性」是错的**：架构级新特性通常先被 gate 在更高的 arch 上（如 Hopper 特性要 `sm_90a`），`-arch=sm_89` 编不过。判据是「`-arch=sm_89` 能否编过，以及 PTX/SASS 是否真出现该指令」，不是显卡代号的宣传词。
- **「PyTorch 能跑 = 我懂 CUDA 语义」是错的**：`torch.cuda.is_available()` 只说明驱动 / 运行时 / toolkit 三者版本自洽，与任何语义承诺无关。
- **版本是耦合的**：驱动 596.49 ↔ toolkit 12.8 ↔ torch 2.11.0+cu128 三者互相约束；单独改一个可能让另一侧失效（例如把 toolkit 换成更高版而驱动不匹配）。

### 语义天花板（本机适用性速查）

| 类别 | 本机可否实机验证 |
|---|---|
| ITS / active mask / 重汇聚（域 01） | ✅ 可（SASS `BSSY/BSYNC/WARPSYNC`） |
| 内存模型 / 原子 / fence（域 04/05） | ✅ 可，但 racecheck 类证据受 admin 阻塞 |
| `cp.async` + 流水（域 09） | ✅ 可（sm_80+） |
| TMA / `cp.async.bulk` / cluster / DSMEM（域 06/09/12） | ❌ 仅概念（sm_90） |
| 128-bit 原子 `.b128`（域 05） | ❌ 仅概念（sm_90） |
| `ncu` / `nsys` 性能计数 | ⚠️ 需管理员；`nsys` 还需先加 PATH |

## 反例

### 反例 A（核心）：拿显卡代号当特性清单

**想当然**：`nvidia-smi` 显示 `compute_cap 8.9`（Ada），既然 Ada 是新架构，那 TMA / cluster 这些「新东西」本机应该也能用。

**事实**：TMA、`cp.async.bulk`、thread block cluster + DSMEM、128-bit 原子都是 **sm_90（Hopper）** 才引入，Ada（sm_89）**没有**。用 `-arch=sm_89` 编译这些指令会被 `ptxas` 拒绝（本机实测见「本机验证」的探针表）。

**判据**：与域 12 的 `EXP-0012` 一致。`cp.async`（本机有）与 `cp.async.bulk` / TMA（本机无）要分开看。证据只有三样：**arch 后缀、编译是否通过、SASS/PTX 是否真的出现该指令**。显卡代号不算证据。

### 反例 B：把「组件存在」当成「可以跑」

**想当然**：`compute-sanitizer.exe` 就在磁盘上，`--tool racecheck` 直接就能验证竞争。

**事实**：本机实测需管理员 / debugger 接口权限才生效；否则 `compute-sanitizer` 早退、`ncu` 报 `ERR_NVGPUCTRPERM`。于是域 04/06/11 只能靠数值证据 + PTX/SASS + 语义定义，做不到「sanitizer 直接抓竞争」这一步。

**教训**：基线要记的是「**在本机、以当前权限，能实际产出什么**」，不是「装了哪些 exe」。

## 本机验证（命令 + 观察）

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
& 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe' --version
& 'E:\venvs\pytorch312\Scripts\python.exe' -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
Test-Path 'E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe'
```

本机观察（2026-10-11，上面四条命令的实际输出；nvcc 的 banner 只保留版本行）：

```
name, driver_version, memory.total [MiB], compute_cap
NVIDIA GeForce RTX 4060 Laptop GPU, 596.49, 8188 MiB, 8.9
Cuda compilation tools, release 12.8, V12.8.61
2.11.0+cu128 12.8 True NVIDIA GeForce RTX 4060 Laptop GPU
True
```

**天花板门（可选，证 sm_89 的上限）**：对使用 TMA / `cp.async.bulk` 的 `.cu` 执行
`nvcc -arch=sm_89 -cubin` 会**编译失败**；换成 `-arch=sm_90a` 才通过。这条「编不过」正是本机语义边界的直接证据（与域 12 的 `cpasync_gate` 同源思路）。

编译命令（路径含空格，`call` 必须加引号）：

```bat
call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -arch=sm_89 -cubin -o out.cubin gate.cu
```

2026-10-11 实测（每个探针都是独立的 `.cu`，只做编译、不运行；`ptxas` 报错即编不过）：

| 探针 | `-arch=sm_89` | `-arch=sm_90a` | 说明 |
|---|---|---|---|
| `cp.async.ca`（对照） | 通过（exit 0） | — | sm_80+，本机有 |
| `cp.async.bulk` + mbarrier | 失败：`Feature 'cp.async.bulk' requires .target sm_90 or higher` | 通过 | TMA 类 bulk copy |
| `cp.async.bulk.tensor`（TMA 张量） | 失败：同上（`cp.async.bulk.tensor` requires sm_90） | 通过 | 真正的 TMA |
| `%cluster_ctaid`（cluster） | 失败：`Feature '%cluster_ctaid' requires .target sm_90 or higher` | 通过 | thread block cluster |
| `atom.global.exch.b128`（128-bit 原子） | 失败：`Instruction 'atom.exch.b128.global' requires .target sm_90 or higher` | 通过 | 域 05 的 `.b128` |
| fp8 `mma.sync` e4m3 | sm_80 失败：`requires .target sm_89 or higher`；sm_89 通过 | — | fp8 tensor core 从 sm_89 起 |

运行探针（1 block × 32 线程，`probe_rt.cu`，编译为 `-O3 -lineinfo -arch=sm_89`）：

```
name=NVIDIA GeForce RTX 4060 Laptop GPU cc=8.9 sm_count=24 warpSize=32 totalGlobalMem=8187 MiB
kernel_out=42 sync=no error
```

工具权限（同机、非管理员，`IsInRole(Administrator) = False`）：

```
compute-sanitizer --tool memcheck probe_rt.exe
========= Error: process didn't terminate successfully
========= Error: Target application terminated before first instrumented API call
ncu --metrics sm__cycles_elapsed.avg probe_rt.exe
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. …
```

版本核对：compute-sanitizer `2025.1.0.0`、ncu `2025.1.0.0`、Nsight Systems `2024.6.2.225`（`C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\…\nsys.exe --version`，`nsys` 不在 PATH）。

## 出处

- *CUDA C++ Programming Guide* — **Compute Capabilities** 特性表（sm_89 vs sm_90：TMA / cluster / `cp.async.bulk` 的门槛）。
- *PTX ISA* — **Target ISA Notes**（各指令的最低 sm 要求；`.b128` 原子、`cp.async.bulk`、cluster 相关指令）。
- *CUDA C++ Programming Guide* — **Independent Thread Scheduling**（Volta+ 起 warp 不再锁步）。
- 本机：CUDA Toolkit 12.8 / nvcc V12.8.61、Nsight Compute 2025.1.0、Nsight Systems 2024.6.2、compute-sanitizer 2025.1.0（版本以本页表格为准）。

## 未决

- **admin 阻塞未解**：`compute-sanitizer` / `ncu` 需要管理员或 `EnableDebuggerInterface.bat` / `RmProfilingAdminOnly=0`。解掉后才能给域 04/06/11 补 racecheck / synccheck / initcheck 证据。
- **驱动 ↔ toolkit 最低版本**：驱动 596.49 对 CUDA 12.8 的最低要求未逐条核对。
- **环境探针的常态化**：已在 `experiments/EXP-0000-baseline/` 提供一键脚本；后续可考虑把它的输出存档作为「每次大改动前后」的基线快照。
