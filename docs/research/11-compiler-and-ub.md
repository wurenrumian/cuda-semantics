---
title: 编译器行为与 UB
type: research
status: draft
created: 2026-10-08
updated: 2026-10-12
tags: [cuda, semantics, compiler, undefined-behavior, restrict, volatile, sm_89]
related:
  - docs/plan/cuda-semantics-topic-map.md
  - docs/research/04-memory-consistency-model.md
  - docs/research/05-atomics.md
  - docs/research/07-ptx-instruction-level.md
  - docs/research/08-floating-point.md
---

# 11 — 编译器行为与 UB

> 域内定位：`docs/plan/cuda-semantics-topic-map.md` §11。
> 目标机：**sm_89（Ada, RTX 4060 Laptop）**，**CUDA 12.8**，`nvcc V12.8.61`。
> 实验源码：`experiments/EXP-0011-compiler-and-ub/scripts/`（`d11_demos.cu`、`run.ps1`）；产物：`tmp/outputs/D11/`（`sass-O3.txt`、`sass-G.txt`、`run-*.txt` 等，本机、未入库）。

## 一句话

CUDA C++ 编译器只对"符合语言语义的程序"做承诺；你一旦越过语义（别名承诺、数据竞争、未初始化、越界），编译器就有权把"这些不可能发生"当作公理去优化——**同一份源码在 `-O3` 下可能给出与 `-G` 相反的结果，这不是编译器 bug，是未定义行为（UB）的定义本身**。

## 先分家：四种"结果怪"只有一种是 UB（也是编译器有权改变语义的）

| 类别 | 例子 | 谁的锅 | 怎么排除/证实 |
|---|---|---|---|
| ① 语言/CUDA 模型 UB | 别名违约、数据竞争、越界、读未初始化 | **调用方**（源码违反语义） | 规范原文 + 语义分析；工具（racecheck/memcheck/initcheck）抓"擦边现场" |
| ② 违反 API 契约 | 传错参数把 null 当合法值、`cudaSetDevice` 后用错 device 指针 | 调用方 | API 错误码（域 10 的观察点表） |
| ③ 编译器优化结果 | 死代码删除、指令调度、`-O3` vs `-G` 差异 | **不是谁的锅**——语义许可范围内的合法变换 | PTX/SASS 对照（域 07 的方法），对照规范确认"仍算的等值" |
| ④ 硬件偶发 | 竞争下的"看起来错/碰巧对"、未定义时序、bit 随机 | 无主（未定义区间的任意实现） | 多次运行 + 语义定性；**不可作为**规范结论 |

调试顺序按 ④→③→②→① 从怪现象往回追：先多轮运行排除 ④ 的随机性，再对照
PTX/SASS 确定 ③ 只是实现选择，再核对 API 契约排除 ②，最后落在"语义违反"的 ①——

**别把所有怪现象都叫 UB**：③ 类（优化结果）在语义上是**合法的**，把它误判成"编译器出 bug"
或"必须关优化"是常见死胡同（见下文"-G 修复错觉"）；④ 类的"偶现对错"恰恰是 ① 的
**常见伪装**——稳定复现的错误反而好判。

## 承诺与边界

编译器把源码编到 SASS 的过程中，做两件事：**保住语义**，和**利用"语义允许它假设的东西"**。域 11 的全部坑都在后者的边界上。

| 语言设施 | 它承诺什么 | 它**不**承诺什么 / 边界 |
|---|---|---|
| 默认（无 `__restrict__`）指针 | 编译器保守假设指针可能别名：对同一地址的访存保持程序顺序 | 需要它激进优化时可能不肯；性能代价不是语义 |
| `__restrict__` / `const __restrict__` | 告诉编译器"这些指针不别名" → 允许 CSE、重排、把 load 换成只读路径（PTX `ld.global.nc`、SASS `LDG.E.CONSTANT`） | **若两个受限指针实际指向同一对象，程序 UB**。编译器可以合法地把 load 提到 store 之前 |
| `const`（只读限定） | 该路径不被写 → 只读缓存路径 | 不提供同步；别名仍按是否 `__restrict__` 判定 |
| `volatile` | 访问**不被编译器消除/合并**，相邻 volatile 访问不重排；SASS 落成 `LDG/STG.E.STRONG.SYS` | **不是原子原语，也不是同步原语**：`*c = *c + 1` 仍是 LD→IADD→ST 三步，多线程会丢更新；也不建立 happens-before |
| 原子（`atomicAdd` 等） | 单次操作为原子 | 默认是 **relaxed**，不提供顺序/可见性（域 05） |
| fence（`__threadfence*`） | 顺序 + 跨作用域可见性 | 不提供原子性（域 04/06） |
| 无数据竞争的程序 | 结果可复现 | 一旦有并发冲突且无 happens-before，**整个执行 UB**，不是"读到旧值" |
| 不读未初始化内存 | 结果确定 | 读 `cudaMalloc` 后未写过的设备内存 = UB（值任意） |
| 访存在界内 | 结果确定 | 越界读/写 = UB，且未必立即崩（域 10 的错误传播边界） |

三条最容易记错的边界：

1. **UB 是"全程序无意义"，不是"坏值"**。数据竞争的判定**拆成三个都满足才算**的条件：
   (a) 存在**冲突**访问（同一地址、至少一个是写）、(b) **不同线程**、(c) 访存**非原子**且
   **没有** happens-before 排序（Programming Guide §5.7.4）。三缺一就不是竞争——"都读"
   不冲突；(a) 同一线程顺序读写不冲突；(c) 一侧换成 `atomicAdd`（即使 relaxed）就把"非原子"
   拆了；有 barrier/fence/事件的把"没有 happens-before"拆了。CUDA 个别访问还有附加规则
   （如 volatile 被定性为相当 system-relaxed），边缘案例回规范原文对，别只背三条件。
2. **`volatile` 与 `atomic` 是两件事**。`volatile` 挡的是编译器，`atomic` 挡的是硬件并发。两者都不能单独替代 fence。官方明确写：`volatile` 不适合线程间同步。按语境拆开记忆——
   - **device 侧**：`volatile int*` 的共享声明/自旋标志 = "每条读写都真实发生，但**没有**原子性、
     **没有** happens-before、**没有**充当 device fence 的资格"（域 06/07 反复复现）。
   - **host 侧**：C++ 的 `volatile` 在普通 host 代码里的语义**更少**（连指令约束的形态都依赖
     实现）——CUDA C++ 附录专门提醒把 host 端老习惯（`volatile` 忙等）搬进 device 的想法
     直接算错档：应该用 `cuda::atomic_ref` 或原子 intrinsic。
3. **`__restrict__` 是一份"我保证不别名"的合同**，不是性能开关。签错合同，代价是结果错误（见反例一）。

## 反例

### 反例一（主）：`__restrict__` 撒谎 → `-O3` 与 `-G` 结果相反

"我给两个指针都加了 `__restrict__` 让它跑快点" —— 如果这两个指针其实指向同一块内存，就是 UB，而优化器会**真的**利用它。

```cpp
// experiments/EXP-0011-compiler-and-ub/scripts/d11_demos.cu
__global__ void store_then_load_restrict(int* __restrict__ p,
                                         int* __restrict__ q, int* out) {
  q[0] = 42;          // 通过 q 写
  int a = p[0];       // 通过 p 读；__restrict__ 说 p、q 不别名
  out[0] = a;
}
__global__ void store_then_load_plain(int* p, int* q, int* out) {   // 同代码，无 restrict
  q[0] = 42;
  int a = p[0];
  out[0] = a;
}
```

用 `p == q`（同一块 device 内存，初值 7）调用：

```
$ d11_demos.exe restrict          # 即 -O3
  __restrict__ version -> out = 7    # 读到的是 store 之前的值
  plain        version -> out = 42   # 保持了程序顺序
$ d11_demos_G.exe restrict        # 即 -G -O0
  __restrict__ version -> out = 42   # 未优化时反而是"对的"
```

同一份 `__restrict__` 内核，`-G` 给 42、`-O3` 给 7，且 `-O3` 下多次运行稳定复现。**"想当然会错"的点就在于：越优化越错，而且错得完全确定。**

优化器是怎么"合法地"改写读写顺序的——别名关系与两版时序对照：

```
调用方产生的真实别名关系(编译器看不到,只看到 restrict 承诺):

        p ──┐
            ├──▶ 同一块 device 内存 [7]
        q ──┘

程序顺序(restrict 版,承诺约束下):        优化后的实际变化:

  ① ST q[0] = 42   (写)                    LDG p[0](只读路径,提前)
  ② LD p[0]        (读,读到?)              STG q[0] = 42
  ③ ST out[0] = a                          STG out[0] = a(旧值 7)
```

而**通用版（无 `__restrict__`）**没有这份假设，load 被排在 store 之后（观察 A 的
`STG→LDG` 两连），结果保持 42——`restrict` 提供的是"可假设不别名"的**许可**，-O3 才真实
消费这份许可。差别在 `SASS` 层立现，但责任在源码的合同上。

责任链：**调用方违约**（两指针同指一块内存却都声明 restrict）→ 编译器按承诺出牌
（CSE/重排/只读路径）→ 观察到"旧值"。既不是优化器"随机重排"，也不是"GPU 缓存不刷新"；
域 07 反例 1 把同一根问题拆到了指令层。

## 如何判断"是不是优化改了结果"——五步对照法

遇到"关了优化就是对的，开了就错"的现象，按这个梯子走（本文反例一就是标准走法）：

1. **先怀疑 UB，别先怀疑编译器**：源码里有没有别名/竞争/越界/未初始化？（四分家里的 ①）
   有嫌疑时读规范对应条款，确认违约事实链。
2. **对照 `-O3` vs `-G`**：两版都跑，拿到"对值/错值"两组数。注意这**只是采样了两个不同的
   设计点**，`-G` **不是正确性修复**——它加调试钩子、关所有回收，**改变了代码本身**；
   "关掉某开关后结果好了"不等于"该开关在替你执行正确性义务"（`-fmad=false`、`--ftz`
   也可能顺带改指令选择，道理相同）。
3. **看 PTX**：中间层变换最可读（观察 C 的 unused_load 在 PTX 层就已无 load；
   `ld.global.nc` 是否被引入）。
4. **看 SASS**：确认 load/store 的实际顺序与遍数（观察 A 的对照即此步）；区分
   "被删的代码"与"未执行的代码"（域 07 读反汇编指南的要点 4）。
5. **回规范收尾**：第一步若确认违约，三四步看到的即**许可范围内**的改变。写结论时按
   "规范结论 / 编译器观察 / 硬件观察 / 性能测量"分层标注，别把实现观察写成承诺。

## 本机验证（命令 + 观察）

### 环境与编译

```bat
call "E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
:: 在 experiments\EXP-0011-compiler-and-ub\scripts\ 下执行
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -lineinfo -arch=sm_89 -o d11_demos.exe   d11_demos.cu
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -G  -O0      -arch=sm_89 -o d11_demos_G.exe d11_demos.cu
E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe -O3 -arch=sm_89 -ptx -o d11_demos.ptx d11_demos.cu
E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe -sass d11_demos.exe > sass-O3.txt
```

一键复现：`pwsh -File experiments\EXP-0011-compiler-and-ub\scripts\run.ps1`（产物写入 `tmp/outputs/D11/`）。

### 观察 A：`__restrict__` 让 load 跑到 store 前面

`-O3` 下 `store_then_load_restrict` 的 SASS（`sass-O3.txt`）：

```
/*0040*/  LDG.E.CONSTANT R5, [R4.64] ;   // p[0] 先读，走只读路径
...
/*00a0*/  STG.E [R2.64], R9 ;            // q[0]=42 后写
```

对照 `store_then_load_plain`（无 `__restrict__`）：

```
/*0070*/  STG.E [R2.64], R9 ;            // 先写
/*0080*/  LDG.E R5, [R4.64] ;            // 后读 → 读到 42
```

对应 PTX（`d11_demos.ptx`）里，`__restrict__` 版本把普通 load 变成了**非相干只读路径** `ld.global.nc`：

```asm
st.global.u32      [%rd6], %r1;   // q[0]=42
ld.global.nc.u32   %r2,    [%rd5]; // p[0]：.nc = non-coherent 只读，可缓存
```

`-G`（`sass-G.txt`）下则退化为通用 `ST.E` 后接 `LD.E`，顺序保留，所以结果是 42。**结论：`__restrict__` 不只是"省一次访存"，它改变的是编译器对别名与可见性的整个假设。**

### 观察 B：`volatile` 不是原子（丢失更新）

```
$ d11_demos.exe counter-volatile
volatile counter : expected=51200000  observed=200004  lost=50999996
$ d11_demos.exe counter-atomic
atomic   counter : expected=51200000  observed=51200000  lost=0
```

256 线程 × 200000 次 `*c = *c + 1`（`volatile int*`），丢了 99.6% 的更新。`volatile` 只保证"每次读写都真的发生、不被优化掉"，**不保证读-改-写是一个原子整体**。PTX 里是 `ld.volatile.global` + `st.volatile.global` 两条独立指令。

### 观察 C：死存储 / 无用 load 被优化器直接删除

`dead_store`（`out[i]=x; out[i]=x+1;`）在 `-O3` 的 SASS 里只剩一条 store：

```
/*00a0*/  STG.E [R4.64], R7 ;   // 只保留了 x+1 那次；第一次写被判定为 dead
```

`unused_load`（`int x=in[i]; out[i]=7;`）在 **PTX 层就已无 load 指令**：

```asm
mov.u32 %r5, 7;
st.global.u32 [%rd4], %r5;   // 读 in[i] 完全消失
```

把指针换成 `volatile` 后（`dead_store_volatile`），两条 store 都以 `STG.E.STRONG.SYS` 保留。这正说明：**普通访存是可以"不发生"的**；要让访问发生，用 `volatile`（或让结果可观测）。

### 观察 D：数据竞争 = UB（工具判词，2026-10-12 补齐）

设计意图是用 `compute-sanitizer` 直接证明"无同步计数是竞争"。两个维度都已拿到工具判词
（产物 `tmp/outputs/sanitizer-2026-10-12/d11-*`）：

- **racecheck `counter-shared`（本 demo 的 shared 池计数）**：
  `Race reported … volatile_shared_counter+0x410 (d11_demos.cu:50)`，共 **16 hazards（16 errors）**。
  volatile 计数器的竞争被工具点名——与 01 反例 C（RAW ×128）、05 `race-naive`（hazard 1 条）
  构成同族三证。
- **initcheck `uninit`（读 `cudaMalloc` 未写过的内存）**：
  `Uninitialized __global__ memory read`，点名 `d11_demos.cu:106`，共 4 errors。
  本机"读未初始化稳定得 0"的侥幸（见运行记录）被 initcheck 逐条实录——**"UB 碰巧正常"**
  在工具层现形。

`counter-volatile`（global 内存丢更新）本身不在 racecheck 视野（racecheck 只查
shared memory hazard，04/05 已校准），它的证据仍是数值丢失 + `ld/st.volatile` 两条独立指令。

### compute-sanitizer 工具环境（历史现场存档）

> **2026-10-11/12 更新**：下述"本机不可用"根因已查明为 **Windhawk**（进程 mod 注入平台）
> 与 sanitizer 注入撞车；停 Windhawk 服务后所有工具正常（见域 0 与过程记忆
> `tmp/docs/memory/windows-toolchain-pitfalls.md`）。上面观察 D 的判词即解锁后补跑的。
> **重启后自启**——每次跑 sanitizer 前需再停（或配置进程排除）。

解除前本域的诊断记录（保留作"注入能进非 CUDA 进程、独独崩在首个 CUDA 调用"的现场）：

- 位置：`E:\NVIDIA\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe`，版本 `2025.1.0 (build 35351055)`。
- 对**任何 CUDA 目标**（含一个只做 `cudaMalloc` 的最小程序）运行都失败：

  ```
  ========= COMPUTE-SANITIZER
  ========= Error: process didn't terminate successfully
  ========= Error: Target application terminated before first instrumented API call
  ```

- 对照实验：同一 sanitizer 注入 `cmd.exe`（无 CUDA 调用）**正常**（能打印 `injected-ok`），说明注入机制本身可用；崩溃发生在目标程序的**第一次 CUDA 调用**处（用带 `fopen` 埋点的最小程序验证：仅写出 `start`，`cudaMalloc` 之前即退出）。
- 试过无效：`--target-processes all`、`--require-cuda-init no`、`--print-level info`、`--log-file`、`--injection-path`、静态 `-cudart static`、从 sanitizer 目录启动、用 `bin\compute-sanitizer.bat`。
- 判断：**compute-sanitizer 2025.1.0 与驱动 596.49 在本机不兼容**（2025.1.0 早于该驱动）。域 04/06/11 想用 racecheck/initcheck/synccheck 的验证因此全部受阻，是待修的阻塞项。

## 出处

**CUDA C++ Programming Guide（12.8）**

- C++ Memory Model 附录，**§5.7.4 Data Races**：数据竞争定义（两个可能并发的冲突动作、至少一个非原子、无 happens-before）→ 未定义行为。同章 §5.7.5 Message Passing 用 block-scope 原子给出竞争/UB 实例。
  `docs.nvidia.com/cuda/cuda-c-programming-guide/` → `05-appendices/cuda-cpp-memory-model.html`
- C++ Language Support 附录（volatile）：明确"CUDA C++ `volatile` 不适合线程间同步；同步请用 `cuda::atomic_ref` / `cuda::atomic` / Atomic Functions"。
  → `05-appendices/cpp-language-support.html`
- C++ Language Extensions 附录（`__restrict__`）：受限指针允许编译器做 CSE、指令重排等激进优化（含 `device_function(const float* __restrict__ ...)` 示例）。
  → `05-appendices/cpp-language-extensions.html`
- nvcc 选项 `-restrict`：断言所有内核指针参数为 restrict。
  → `02-basics/nvcc.html`
- Special Topics → Virtual Memory Management：同一内核中写一个虚拟别名、读另一个别名而不加同步 = 未定义行为（"*B can take on either the previous value or some value in-between"）。
  → `04-special-topics/virtual-memory-management.html`

**PTX ISA**

- **§8 Memory Consistency Model**：定义 volatile/`mmio` 操作类型、scope（`.cta/.gpu/.sys`）、requires-byte-address、morally strong 操作等；volatile 与普通访存的差别在此。
- **§9.7.8 Data Movement Instructions**：`ld` / `st` 的 cache operator（`.ca/.cg/.cs/.lu/.cv/.wt`）、`.volatile`、`.relaxed/.acquire/.release` 修饰；本机 `volatile` 落成 `ld.volatile.global` / `st.volatile.global`。
  `docs.nvidia.com/cuda/parallel-thread-execution/`

## 未决

- **（已解决，2026-10-12）compute-sanitizer（域 04/06/11 的公共阻塞）**：根因 Windhawk
  注入冲突，停服即恢复。本域已拿到 racecheck（counter-shared 16 errors）与 initcheck
  （uninit 4 errors）判决。**遗留校准**：racecheck 对 global 竞争（counter-volatile 形状）
  不适用——仍以数值 + PTX/SASS + 语义定义叠加论证。
- 未覆盖的 UB 点（本机可直接做，尚未做）：
  - 越界访存：03 域已拿到 `memcheck` 的 `Invalid __local__ read` 判词（同族证据，
    `tmp/outputs/sanitizer-2026-10-11/d03-*`）；本域的 global 越界读未单独跑。
  - 未初始化读：`read_uninit` 结果稳定为 `0 0 0 0`（本机 `cudaMalloc` 常给零页），**不能**
    据此认为"未初始化=0"；这一点已被 initcheck 判词补证（观察 D）。
  - 有符号整数溢出 / 移位 UB、`reinterpret_cast` 触发的严格别名、`memcpy` 之外的 type punning。
  - 编译器对"无副作用死循环"的处理（`while(1);` 是否被保留）。
- `-restrict` 全局开关对所有 kernel 指针的影响未测；`-use_fast_math` 与 UB 的交互留给域 08。
- `__restrict__` 触发 `ld.global.nc` 是本机（sm_89/CUDA 12.8）的代码生成选择；不同 ptxas 版本可能不同，语义结论（可重排）不变。
