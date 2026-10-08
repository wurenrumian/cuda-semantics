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
| `barrier` | 只有一半线程到达 `__syncthreads()`（死锁） | 交给 `compute-sanitizer --tool synccheck` |

编译：`vcvars64.bat` → `nvcc -O3 -lineinfo -arch=sm_89 -Xptxas -v`。

## 结果

产物：`../../outputs/D02/2026-10-08/`（`build.log` / `sched.txt` / `occupancy.txt` / `sass.txt` / `sanitizer-blocked.txt`）。

- `sched`：三次运行 `start==blockIdx? NO`、`finish==blockIdx? NO`、且 start 顺序三次互不相同 → block 调度顺序无保证。
- `occupancy`：`occ_plain`(10 regs)=12 blocks/SM(1536 thr)，`occ_regs`(72 regs)=7 blocks/SM(896 thr)，`occ_smem`(16KB)=5 blocks/SM(640 thr)；`__launch_bounds__(128,8)` 把寄存器压到 64（48B spill）换来 8 blocks/SM，而 `__launch_bounds__(256,8)` 因 256×8>1536 被 ptxas **静默忽略**。
- SASS：`threadIdx`→`S2R SR_TID.X`，`blockIdx`→`S2R SR_CTAID.X`，`%smid`→`S2R SR_VIRTUALSMID`，`__syncthreads()`→`BAR.SYNC.DEFER_BLOCKING 0x0`。
- `compute-sanitizer` 在本 opencode 会话无法注入目标进程（"terminated before first instrumented API call"），详见 `sanitizer-blocked.txt`；registry 键已为 1、当前会话非管理员。

## 结论

- **不承诺** block 的调度/完成顺序；依赖"block 0 先跑"是 UB。
- 占用率是线程数、寄存器、shared memory 三者取最小；`__launch_bounds__` 的 `minBlocks` 只是 hint，不可行时被忽略。
- `threadIdx`/`blockIdx`/`%smid` 是特殊寄存器读取，不是内存访问。

## 未决

- 在普通桌面终端复跑 `compute-sanitizer`（`--tool synccheck` / `memcheck`）；必要时管理员终端跑 `EnableDebuggerInterface.bat`。
- cooperative groups `grid.sync()` 的跨 block 同步（留域 6）。

## 运行记录

| 时间 | 命令 | 产物 |
|---|---|---|
| 2026-10-08 | run.ps1（build → sched → occupancy → sass → synccheck） | `../../outputs/D02/2026-10-08/` |
