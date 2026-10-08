# EXP-0002 — 同步原语（域 06）

## 状态

done（2026-10-08）· 见 `tmp/docs/research/06-synchronization.md`

## 问题

1. `__syncthreads()` 的承诺到底是什么？在分支里调用会怎样？
2. `__syncwarp()` 的 mask 语义：默认 `0xffffffff` 在发散代码里安全吗？
3. 命名屏障 `bar.sync id, count` 有什么硬约束？

## 假设

- H1：`__syncthreads()` 只要有一个 warp 没到就会死锁。
- H2：`__syncwarp()` 在 `if (lane<16)` 里调用一定会挂。
- H3：命名屏障的 `count` 任意都行。

## 方法

`scripts/sync_demo.cu` 若干模式 + `scripts/run.ps1`（build → cubin → `cuobjdump -sass` → 各模式 → sanitizer 模板命令）。
一种"退出 vs 存活"的对照：让没到屏障的一半线程要么 `EXIT`，要么自旋（读一个永不变化的 global flag）。
自旋模式由 host 看门狗轮询 `cudaStreamQuery` ~5s 判定是否 wedge。

## 结果

见 `tmp/outputs/D06/2026-10-08/`。

| 观察 | 结果 |
|---|---|
| `divergent-syncthreads`（另一半 EXIT） | **FINISHED**（没挂） |
| `divergent-syncthreads-alive`（另一半自旋） | **WEDGED**（~5s 未完成） |
| `divergent-syncwarp`（另一半 EXIT） | **FINISHED** |
| `syncwarp-partial-alive`（另一半自旋） | **WEDGED** |
| `named-barrier`（`bar.sync 1/2,128` 分半） | 正确 128/128 |
| `bar.sync 1,16` 编译 | ptxas **报错**：count 必须是 warp 的倍数 |
| SASS | `BAR.SYNC 0x0`；命名屏障 `BAR.SYNC 0x2,0x80`；warp 屏障 `WARPSYNC 0xffffffff` |

- H1 **错**：Volta+ 下 `bar.sync` 只等"未退出"的线程；没到的线程若已 EXIT，屏障照样满足。
- H2 **错**（但同因）：mask 里已 EXIT 的 lane 不算数；只有仍存活的 lane 才会把它挂住。
- H3 **错**：`count` 必须是 warp size 的倍数，ptxas 直接拒绝。

## 结论

- 发散调用 `__syncthreads()`/`__syncwarp()` 是 UB；**它在测试里过不过，取决于没到的线程是否已退出** —— 这正是"想当然会错"的点。
- `__syncthreads()` 编成 `bar.sync 0`（无 count = 全 CTA）；命名屏障 0..15，count 必须是 32 的倍数。
- 编译器会在屏障前插 `WARPSYNC`（和 `BRA.CONV`）来强制收敛。

## 未决 / 阻塞

- **`compute-sanitizer` 在本机无法注入**：任何 CUDA 程序（含 trivial）都报
  `Error: Target application terminated before first instrumented API call`。
  同机 `ncu` 能 attach 但计数器报 `ERR_NVGPUCTRPERM`。两者都需要 admin/权限；
  当前用户非管理员。racecheck / synccheck 的观察待有权限环境复核。
  （探针程序与其输出在 `tmp/outputs/D06/2026-10-08/_sanitizer-probe/`。）

## 运行记录

| 时间 | 命令 | 产物 |
|---|---|---|
| 2026-10-08 | `scripts/run.ps1` | `tmp/outputs/D06/2026-10-08/` |
