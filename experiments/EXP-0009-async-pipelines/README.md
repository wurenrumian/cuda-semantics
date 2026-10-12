# EXP-0009 — 异步与流水语义

对应域 09（`docs/research/09-async-pipelines.md`）。

## 状态

done（2026-10-08 首轮）

## 问题

1. kernel launch / `cudaMemcpyAsync` / `cp.async` 的「异步」到底承诺什么、边界在哪？
2. 不同 stream 之间有没有顺序？默认 stream 在两种语义下分别是什么？
3. `cp.async` 的 commit/wait 协议，漏掉 wait 会怎样（SASS 上长什么样）？
4. CUDA Graph 的 capture/replay 语义。
5. sm_89 支持 / 不支持哪些异步搬运指令？

## 假设

- H1：`cudaMemcpyAsync` 只有在 pinned 主机内存上才真异步。
- H2：legacy 默认 stream 与其它 stream 隐式同步；`--default-stream per-thread` 下不。
- H3：`cp.async` 后不 `wait_prior` 就读 shared 是数据竞争（UB）。
- H4：capture 只记录不执行；replay 每次重新执行。
- H5：`cp.async.bulk`（TMA）在 sm_89 被 ptxas 拒绝、sm_90 通过。

## 方法

一个 `async_pipelines.cu`，多模式；另编两份只改 `--default-stream` 的可执行文件；一个
`tma_probe.cu` 探针。见 `scripts/`。`run.ps1` 负责 build → run → SASS dump → TMA 探针 →
compute-sanitizer。

| 模式 | 做什么 | 观察 |
|---|---|---|
| `props` | 打印 device props | asyncEngineCount / managedMemory / pageableMemoryAccess |
| `streams` | 4 个占 1 SM 的 kernel：1 stream vs 4 streams | 墙钟 → 是否重叠 |
| `ds` | 显式 stream + 默认 stream 各一个长 kernel | legacy vs per-thread 的比值 |
| `memcpy` | 256 MiB H2D：sync / pageable async / pinned async | `cudaMemcpyAsync` 返回耗时 |
| `cpasync` | `__pipeline_memcpy_async` + `wait_prior(0)` | 正确性 + SASS `LDGSTS` |
| `cpasync-racy` | 同上但删掉 `wait_prior` | 是否出现错值 + SASS 缺 `DEPBAR` |
| `graph` | capture 两 kernel、replay 两次 | capture 后/replay 后计数 |

## 结果

见 `../../tmp/outputs/D09/2026-10-08/`。

- `memcpy`：pageable **33.19 ms** vs pinned **0.013 ms** → 支持 H1。
- `ds`：legacy 比值 **1.99**、per-thread **1.04** → 支持 H2。
- `cpasync-racy`：首轮 **256/256** 错、其后 0/256；SASS 少一条 `DEPBAR.LE SB0,0x0` → 支持 H3。
- `graph`：capture 后 counter=0，replay 后 2、4；null stream capture 报
  `cudaErrorStreamCaptureUnsupported` → 支持 H4。
- `tma`：sm_89 报 `requires .target sm_90 or higher`，sm_90 通过 → 支持 H5。
- `streams`：4 stream 比 1 stream 快 **4.46×** → 并行的「机会」成立，但不是承诺。
- SASS：`cp.async` → `LDGSTS.E.BYPASS.128`；普通写法 → `LDG.E.128.CONSTANT` + `STS.128`。

## 结论

- 「异步」只保证**入队**，不保证并行、不保证不阻塞主机（pageable `MemcpyAsync`）。
- 顺序只在**同 stream 内**和**显式事件**里成立；legacy 默认 stream 会隐式串起跨 stream 工作。
- `cp.async` 必须 `wait_prior` 后才可读 shared；SASS 差异是一条 `DEPBAR`。
- capture 记录不执行，replay 重放整图。
- TMA / `cp.async.bulk` 是 sm_90 边界，本机不具备。

## 未决

- **（已解决，2026-10-12）racecheck 证据已补**：Windhawk 注入冲突解除后，
  `cpasync-racy` 报 5 组 hazard（写方为 `cp_async` 的 async-proxy 访问），`cpasync` 正确版 0
  hazard；产物 `tmp/outputs/sanitizer-2026-10-12/d09-*`。`ncu` 计数器权限仍待管理员。
- 未跑 `ncu` / `nsys`；未做多 stage double buffering；未测统一内存；未做「铺满 GPU 不重叠」对照。

## 运行记录

| 时间 | 命令 | 产物 |
|---|---|---|
| 2026-10-08 | `scripts/run.ps1`（build → 各模式 → SASS → TMA 探针 → sanitizer） | `../../tmp/outputs/D09/2026-10-08/` |
| 2026-10-12 | racecheck `cpasync` / `cpasync-racy`（0 vs 5 hazards） | `tmp/outputs/sanitizer-2026-10-12/d09-*.txt` |
