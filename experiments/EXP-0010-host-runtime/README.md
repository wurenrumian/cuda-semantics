# EXP-0010 — 宿主侧运行时 / 启动语义

## 状态

done（2026-10-08 首轮）

## 问题

1. `<<<grid, block, smem, stream>>>` 的「启动」到底承诺了什么、什么时候校验？
2. 错误是怎么冒出来的：哪些是同步的、哪些是异步的、`cudaGetLastError` 与
   `cudaPeekAtLastError` 差在哪、异步错误清了能不能恢复？
3. stream 之间的顺序/并发承诺是什么？legacy default stream 的隐式同步能不能观察到？
4. 动态并行（device 侧 `<<<>>>`）的编译/运行边界。
5. `__launch_bounds__` 这种宿主侧约束如何落到设备模块（PTX）。

## 方法

`scripts/runtime_demo.cu` 五个模式（每个都新起进程，因为异步 fault 会污染上下文）：

| 模式 | 做什么 | 观察什么 |
|---|---|---|
| `config`   | `<<<0,256>>>` / `<<<1,2000>>>` / `<<<1,32,200000>>>` | 非法配置是否**当场**报错 |
| `peekget`  | 制造一次 launch 错误，反复 peek / get | peek 不清、get 清 |
| `async`    | `deref_null<<<1,1>>>(nullptr)` | launch 后是否 success；错误何时出现、清了是否还在 |
| `streams`  | 自旋 kernel（default stream，带 `clock64()` 时间上限）+ 写标志位（blocking vs non-blocking stream） | default stream 是否隐式串行 |
| `props`    | 打印 `cudaDeviceProp` 启动相关字段 | 本机硬边界 |

另有：
- `scripts/smem_optin.cu` —— 64 KiB 动态 smem 在 `cudaFuncSetAttribute` 前后的差别。
- `scripts/dp_parent.cu` —— 父 kernel 启动子 kernel（动态并行），验证 `-rdc=true -lcudadevrt` 边界。
- `scripts/launchbounds.cu` —— `__launch_bounds__(256,2)` → PTX `.maxntid` / `.minnctapersm`。

注意：`streams` 的自旋用 `clock64()` 计时超时（约 0.1 s），所以「被隐式串行化」的那一路会在超时后**退出**并报告 `flag=0`，进程不会挂住；「真并发」那一路会在若干微秒内看到 `flag=1` 提前退出。

编译：`vcvars64.bat` → `nvcc -O3 -lineinfo -arch=sm_89`（DP 另加 `-rdc=true -lcudadevrt`）。

## 结果

见 `../../tmp/outputs/D10/2026-10-08/`。

- `config`：三种非法配置都**同步**返回错误（9 / 9 / 1）→ launch 配置在提交时校验。
- `peekget`：peek 连读两次都报同一错误；get 一次后即清空。
- `async`：launch 后 `cudaSuccess`；`cudaDeviceSynchronize()` 才报 `700`（illegal memory
  access），且 get 清掉后再度 sync **仍**报 700 → 粘性、上下文级。
- `streams`：blocking stream 下自旋看到 `flag=0`（被 default stream 隐式串行）；
  non-blocking 下看到 `flag=1`（真并发）。
- `props`：`sharedMemPerBlock=49152`、`sharedMemPerBlockOptin=101376`、`maxThreadsPerBlock=1024`。
- `smem_optin`：64 KiB 在 setattr 前 `invalid argument`，之后成功。
- `dp`：无 `-rdc` → 编译错误 "requires separate compilation mode"；有则跑通（child wrote 42）。
- `launchbounds`：PTX 里出现 `.maxntid 256, 1, 1` 与 `.minnctapersm 2`。

## 结论

- 启动 = **异步提交**：同步只保证「启动描述合法」，执行错误延后到同步点、且粘性。
- launch 配置错误与执行错误走**两条通道**；`cudaGetLastError` 清的是错误槽位，不是上下文。
- 同 stream in-order；异 stream 无顺序承诺；legacy default stream 会显式地把 blocking stream 串起来。
- 动态并行要 `-rdc=true -lcudadevrt`；CUDA 12 默认 CDP2，device 侧 `cudaDeviceSynchronize` 不再可用。

## 未决

- compute-sanitizer 本机无法 attach（连自带 `sanitizer_smoke.exe` 都失败），见讲解稿「未决」。
- 未用 `nsys` 定量验证 non-blocking stream 的真并发。

## 运行记录

| 时间 | 命令 | 产物 |
|---|---|---|
| 2026-10-08 | `run.ps1`（build → config/peekget/async/streams/props → memcheck → smem/dp/ptx） | `../../tmp/outputs/D10/2026-10-08/` |
