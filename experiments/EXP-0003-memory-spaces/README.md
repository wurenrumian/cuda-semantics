# EXP-0003 — 内存空间语义（register / local / shared / global / constant / texture）

## 状态

done（2026-10-08 首轮）

## 问题

1. 六个内存空间各自承诺了**作用域 / 生命周期 / 可见性**的什么？变量到底落在哪个空间？
2. "数组 = local memory = 慢"这种直觉在 sm_89 上成立吗？
3. shared bank conflict、global 合并访问、constant 广播各自的语义边界在哪？

## 假设

- H1：`float arr[16]` 若用**运行时**索引，就会落到 local memory（SASS `LDL/STL`）。
- H2：local 是每线程私有、但物理在 device memory（与 global 同级），不是片上暂存。
- H3：shared 只在 block 内可见、bank conflict 会成倍放慢。
- H4：constant 的"快"只在**同地址广播**时成立，按 lane 分叉会串行化。
- H5：global 合并访问与访问跨度强相关，工作集需大于缓存才看得到。

## 方法

一个 `.cu`（`scripts/mem_spaces.cu`），模式见下表；`scripts/run.ps1` 负责 vcvars → 编译（`-Xptxas -v`）→ 生成 PTX → `cuobjdump -sass` → 各模式计时 → sanitizer。

| 模式 | 做什么 | 观察 |
|---|---|---|
| `attrs` | `cudaFuncGetAttributes.localSizeBytes` | 寄存器 vs local 占用 |
| `local` | `volatile float arr[16]` 动态索引循环 | SASS `LDL/STL`、计时 |
| `chase` | 依赖索引 pointer-chase：local vs shared | 两种空间的 load-to-use 延迟差 |
| `const` | `__constant__` uniform 地址 vs 每 lane 分叉地址 | `ULDC` vs `LDC`、计时 |
| `shared` | 32 lanes，地址 = lane*stride + r，stride∈{1,32} | bank conflict 倍数 |
| `coalesce` | global，stride∈{1,32}，32MB 工作集、65536 线程 | 合并/非合并带宽比 |
| `readonly` | `__ldg` / `const __restrict__` | SASS `LDG.E.CONSTANT` |
| `oob` | 越界读 8-float local 数组 | `compute-sanitizer --tool memcheck` |

编译：`call vcvars64.bat` → `nvcc -O3 -lineinfo -arch=sm_89 -Xptxas -v`。

## 结果

产物在 `tmp/outputs/D03/2026-10-08/`（build.log、`mem_spaces.ptx`、`sass.txt`、各模式 txt、`sass-opcode-histogram.txt`、`ptx-state-spaces.txt`、`sanitizer-memcheck-oob.txt`）。

- **H1 半否定（关键）**：`float arr[16]` 用**可证明在界内**的运行时索引时，ptxas 把整个数组**提升回寄存器**（0 LDL/STL、0 stack）。只有 `volatile`/长循环索引时才真的进 local（64B stack、32 LDL）。PTX 里甚至出现 `st.local`/`ld.local` 而 SASS 一条 `LDL/STL` 都没有——**空间是编译器决定，必须看 SASS**。
- **H2 支持**：local 数组的访问是 `LDL/STL`、`localSizeBytes=64`、有 64B 栈帧（32B 栈帧是 `oob` 模式 8-元素数组 `k_local_oob` 的数字，与 `k_local_loop` 的 16-元素数组无关）；依赖链延迟 local(LDL) 65.3ms vs shared(LDS) 55.4ms（1.18x，被循环开销稀释；2026-10-08 首测 108.1/91.4ms，倍数不变）。
- **H3 支持**：shared stride=1 1.95ms vs stride=32 8.74ms（≈4.5x，32 路 bank conflict；2026-10-08 首测 4.10/20.05ms → 4.9x，绝对值随时钟状态浮动）。
- **H4 支持**：constant broadcast ≈15.2ms vs divergent ≈39.3ms（≈2.6x；首测 35.16/91.55ms → 2.6x）；SASS broadcast=`ULDC`（uniform 通路），divergent=`LDC`。
- **H5 支持**：global stride=1 0.5ms vs stride=32 6.0ms（≈12x，32MB 工作集、65536 线程；2026-10-08 首测 0.36/9.95ms → 27.6x，绝对值随时钟状态浮动、量级不变）；小 buffer 时差距被缓存掩盖。
- **（更新 2026-10-11）sanitizer 已通**：此前注入失败根因是 Windhawk 进程注入冲突（非权限）；停 Windhawk 后 `memcheck oob` 正面报出 `Invalid __local__ read ... out of bounds`，产物 `tmp/outputs/sanitizer-2026-10-11/d03-memcheck-oob.txt`。`ncu` 计数器权限（ERR_NVGPUCTRPERM）仍待管理员。

## 结论

- "内存空间"要先分清是**作用域/生命周期**语义，还是**性能/物理落点**语义；后者在 sm_89 上是**编译器 + 架构**共同决定，且可在 PTX/SASS 里核对。
- 最典型的"想当然会错"：源码写 `float arr[N]` ≠ 它在 local memory；动态索引 forcing local 也**未必**成立（可能被提升回寄存器）。
- constant 的"广播快"、shared 的"比 global 快"都有边界（分叉 / bank conflict）。

## 未决

- compute-sanitizer 注入失败问题已在 2026-10-11 解决（Windhawk 注入冲突；见 doc 03「观察 7」与 `tmp/outputs/sanitizer-2026-10-11/`）。`ncu` 计数器权限仍待管理员。
- ncu bank-conflict / L1 指标需 `RmProfilingAdminOnly=0` 或管理员。
- texture object / sampler 路径未做（仅覆盖 `__ldg` 只读通路）。
- register→local spill 阈值、L1/共享内存 carveout 归域 2/12。

## 运行记录

| 时间 | 命令 | 产物 |
|---|---|---|
| 2026-10-08 | `pwsh -File scripts/run.ps1` | `tmp/outputs/D03/2026-10-08/` |
