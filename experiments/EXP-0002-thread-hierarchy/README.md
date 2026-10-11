# EXP-0002 — 线程层次与调度（block 顺序 / occupancy / `__launch_bounds__`）

## 状态

done（2026-10-08；sanitizer 一项受阻，见下）

## 问题

1. grid 里的 block 是否按 `blockIdx` 顺序启动/完成？跨运行是否稳定？
2. 一个 SM 上能同时驻留多少 block？由哪些资源决定？
3. `__launch_bounds__(maxTPB, minBlocks)` 到底做了什么？有没有"想当然会错"的地方？
4. `threadIdx` / `blockIdx` / `%smid` / `__syncthreads()` 在 SASS 里长什么样？

## 方法

一个 `.cu`，三种模式（见 `scripts/thread_hierarchy.cu`）：

| 模式 | 做什么 | 观察什么 |
|---|---|---|
| `sched` | 1024 block × 128 线程；每个 block 用全局 `atomicAdd` 拿"启动票"和"完成票"，并记录 `%smid` | 启动/完成顺序 vs `blockIdx` 顺序；三次运行是否一致 |
| `occupancy` | 5 个 kernel（普通 / 高寄存器 / `__launch_bounds__` 两种 / 大 shared） | `cudaDeviceProp`、`cudaFuncGetAttributes`、`cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| `barrier` | 两个**有界**的块内同步演示：① 32 线程全部到达 `__syncthreads()`，其中 0–15 号先做定量的活；② 64 线程的 block 里只用命名屏障 `bar.sync 1, 32` 同步 warp 0 | `__syncthreads()` 先到的线程等多久；命名屏障的 `count` 语义 |

编译：`vcvars64.bat` → `nvcc -O3 -lineinfo -arch=sm_89 -Xptxas -v`。

> **关于 `barrier` 的改动（2026-10-11）**：原来这一模式跑的是"只有一半线程到达 `__syncthreads()`"的死锁 kernel，
> 里面用 `while (true) __nanosleep(1000);` 把另外一半线程永远停在屏障之前。那条路径**不会自行结束**：
> `thread_hierarchy.exe barrier` 会永久挂住，`cudaDeviceSynchronize()` 等不到返回。
> 由于本机 `compute-sanitizer` 注入失败（见下），没有工具能替它收尾，所以：
> - kernel 里不再有**无界自旋**；`barrier` 改成上面两个**定量、必然结束**的演示；
> - 真正的"一半线程永不到达"写法只在本文档里作为反例保留（见下），不提供可跑模式；
> - `run.ps1` 给每次运行被测程序（sched / occupancy / barrier / sanitizer）加了 host 端看门狗（超时即连同进程树 `taskkill /T` 杀掉并在输出里写 `[WATCHDOG]`），
>   所以即使将来某个模式真的卡住，脚本和 runner 也不会被拖死。

死锁反例（**不要单独运行**，除非外面套了超时）：

```cuda
__global__ void barrier_deadlock(int *out) {
  if (threadIdx.x < 16)
    while (true) __nanosleep(1000);  // 这 16 个线程永远到不了屏障
  __syncthreads();                   // 另外 16 个线程在这里等到天荒地老
  out[threadIdx.x] = 1;
}
```

`__syncthreads()` 等价于整块的 `bar.sync 0`，**不接受参与者数量**；只有一部分线程到达就是未定义行为，
实际上就是永久等待。想只同步一部分线程，要用带 `count` 的命名屏障，而 `count` 必须是 warp 大小的整数倍
（PTX ISA §9.7.13.1），所以"32 个里同步 16 个"根本表达不出来。

## 结果

产物：`tmp/outputs/D02/2026-10-08/`（`build.log` / `sched.txt` / `occupancy.txt` / `barrier.txt` / `sass.txt` / `sanitizer-blocked.txt`）。

- `sched`：三次运行 `start==blockIdx? NO`、`finish==blockIdx? NO`、且 start 顺序三次互不相同 → block 调度顺序无保证。
- `occupancy`：`occ_plain`(10 regs)=12 blocks/SM(1536 thr)，`occ_regs`(72 regs)=7 blocks/SM(896 thr)，`occ_smem`(16KB)=5 blocks/SM(640 thr)；`__launch_bounds__(128,8)` 把寄存器压到 64（48B spill）换来 8 blocks/SM，而 `__launch_bounds__(256,8)` 因 256×8>1536 被 ptxas **静默忽略**。
- `barrier`：32 线程的 `__syncthreads()` 里，晚到的 0–15 号花掉的时间就是早到的 16–31 号在屏障上等的时间，最后 32 条线程在同一个时钟点被放行；命名屏障 `bar.sync 1, 32` 只让 warp 0 参与，warp 1 自始至终不被阻塞。
- SASS：`threadIdx`→`S2R SR_TID.X`，`blockIdx`→`S2R SR_CTAID.X`，`%smid`→`S2R SR_VIRTUALSMID`，`__syncthreads()`→`BAR.SYNC.DEFER_BLOCKING 0x0`，命名屏障→`BAR.SYNC.DEFER_BLOCKING 0x1, 0x20`（障碍 1 号，32 个参与者）。
- `compute-sanitizer` 在本 opencode 会话无法注入目标进程（"terminated before first instrumented API call"），详见 `sanitizer-blocked.txt`；registry 键已为 1、当前会话非管理员。

## 结论

- **不承诺** block 的调度/完成顺序；依赖"block 0 先跑"是 UB。
- 占用率是线程数、寄存器、shared memory 三者取最小；`__launch_bounds__` 的 `minBlocks` 只是 hint，不可行时被忽略。
- `threadIdx`/`blockIdx`/`%smid` 是特殊寄存器读取，不是内存访问。
- `__syncthreads()` 要求块内**所有**未退出的线程都到达；要同步一个子集只能用带 `count` 的命名屏障，且 `count` 必须是 warp 大小的整数倍。

## 未决

- 在普通桌面终端复跑 `compute-sanitizer`（`--tool synccheck` / `memcheck`）；必要时管理员终端跑 `EnableDebuggerInterface.bat`。
- cooperative groups `grid.sync()` 的跨 block 同步（留域 6）。

## 运行记录

| 时间 | 命令 | 产物 |
|---|---|---|
| 2026-10-08 | run.ps1（build → sched → occupancy → sass → synccheck） | `tmp/outputs/D02/2026-10-08/` |
| 2026-10-11 | run.ps1（加了逐进程看门狗；新增 bounded `barrier` 演示） | `tmp/outputs/D02/2026-10-08/` |
| 2026-10-11 | run_dyn_smem.ps1（动态 shared memory × launch_bounds 同池验证） | `tmp/outputs/D02/2026-10-11/dyn-smem/` |
