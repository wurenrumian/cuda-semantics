---
title: 域 05 — 原子语义
type: research
status: draft
created: 2026-10-08
updated: 2026-10-12
tags: [cuda, semantics, atomics, memory-model, sm_89]
related: [docs/plan/cuda-semantics-topic-map.md, experiments/EXP-0005-atomics, tmp/outputs/D05]
---

# 域 05 — 原子语义

> 目标机：RTX 4060 Laptop（**sm_89 / Ada**）、CUDA 12.8（nvcc V12.8.61）、driver 596.49。
> 实验代码：`experiments/EXP-0005-atomics/scripts/`；产物：`tmp/outputs/D05/`。出处见文末。

## 一句话

CUDA 的原子只承诺**「在某个作用域内，一次读-改-写不可分割」**；默认 intrinsic（`atomicAdd` 等）是
**relaxed 顺序 + device 作用域**，它**不**附带任何内存序、不充当屏障、不建立 happens-before——
「原子」和「有序」是两件正交的事。

## 与域 04 的分工

04 讲的是**内存模型的通用规则**(读写彼此如何"对齐看":happens-before、order、scope 的语义底座);
05 讲的是**原子这个 API 家族**(哪些操作能当"最小不可分步"用、每个接口怎么传 scope/order、
怎么用 CAS 自己搭任意 RMW)。关系是"04 给语义底座,05 给使用姿势":

```
        ┌─────────────────────────────────────────────┐
        │  域 04 ·内存模型(通用规则)                  │
        │  order(relaxed/acquire/release…) 定义       │
        │  ——"这一次读写相对谁有序"                   │
        │  scope(cta/gpu/sys) 定义                   │
        │  ——"这个保证对谁生效"                      │
        └──────────────────┬──────────────────────────┘
                            │ 作为参数被「原子 API」消费
                            ▼
        ┌─────────────────────────────────────────────┐
        │  域 05 ·原子 API 家族(本篇)                │
        │  原子性 = 一次 RMW 不可分割(永远有)         │
        │  order/scope = 可选增强(按需传参)           │
        │  自建任意 RMW = atomicCAS 循环              │
        └─────────────────────────────────────────────┘
```

两篇会重复出现同一批词(order/scope/fence),但 04 把它们当**定义**讲,本篇把它们当**参数**讲——
看 05 的反例时,判定规则一律回到 04(尤其 04「四概念大白话」与「观察 3.5」的 SASS 落点表)。

CUDA 的原子只承诺**「在某个作用域内,一次读-改-写不可分割」**;默认 intrinsic(`atomicAdd` 等)是
**relaxed 顺序 + device 作用域**,它**不**附带任何内存序、不充当屏障、不建立 happens-before——
「原子」和「有序」是两件正交的事。

## 原语地图:先分清功能,再谈参数

把"原子"四个功能群分开,而不是一切从 `atomicAdd` 出发:

| 功能群 | 干什么 | 传统 intrinsic(默认 relaxed + device,无参数) | cuda::atomic / atomic_ref(scope/order 可传) |
|---|---|---|---|
| **RMW**(不可分地改) | 加/减/min/max/位运算/交换 | `atomicAdd` `atomicSub` `atomicMin` `atomicMax` `atomicAnd/Or/Xor` `atomicExch` | `fetch_add/or/xor/max…`、`exchange` |
| **条件更新**(可自建任意 RMW) | 期望值仍相等才写 | `atomicCAS` | `compare_exchange_weak/strong` |
| **原子读/写** | 单次读/写，可携带内存序 | (旧式无对应物;要靠 `volatile` + fence 凑,不推荐) | `load` / `store` |
| **fence**(只排序,不原子) | 给前后内存操作定型 | `__threadfence_block/_device/_system` | `cuda::atomic_thread_fence` |

注意第一行的"不可分地改"与第四行的"只排序不原子"是**两种不同保证**:RMW 永远不可分割,但 relaxed
的 RMW 不排任何别的东西;fence 恰好相反,不针对任何对象,只约束它前后的操作。

### 速查表：返回值、成败判定与支持类型

（类型明细为本机 header 实测：CUDA 12.8、sm_89，见 §本机验证；**传统 intrinsic 与 libcu++
接口的差别在"成败/返回值怎么给"与"参数往哪放"两处，容易互相套错**。）

| 接口 | 本机支持类型 | 返回值 | 成功/失败怎么判 | order/scope 放哪里 |
|---|---|---|---|---|
| `atomicAdd/Sub/Min/Max/And/Or/Xor/Exch(p, v)` | 整型 32/64；`float`（仅 Add，sm_20+）；`double`（仅 Add，sm_60+） | 提交前的旧值 | 没有"失败"分支——RMW 无条件提交 | **没有参数**：默认 relaxed + device；scope 用 `_block`/`_system` **后缀** |
| `atomicCAS(p, cmp, val)` | 整型 32/64 | 提交瞬间 `*p` 的值 | **返回值不区分成败**：`== cmp` 即成功，否则失败且未写 | 同上（无 order 参数） |
| `cuda::atomic_ref<T, S>` `load/store(order)` | 任意标量（宽度取决于架构，`atomicity` 由编译产物保证） | load 返回值；store 无返回 | 无失败分支 | order 是**函数参数**，scope 是**模板参数** |
| `compare_exchange_strong/weak(expected, desired, …)` | 同上 | **bool**：成败 | 失败时 `expected` 被**自动更新为当前值**，直接拿去重试 | 同上（成败一参一函数） |
| `fetch_add/or/max/…(v, order)` | 同上 | 提交前的旧值 | 无失败分支 | 同上 |

三件容易踩的接口差别：① 传统 `atomicCAS` 失败也**返回值**（送你情报但不给结论），C++ 的
compare-exchange 直接给结论还顺手更新 `expected`——别把两边的"重试循环"写串。② **不要把
`atomicAdd(float)` 的存在推广成"原子全家桶都有重载"**：`Min/Max/Exch/CAS` 的浮点重载本机
（sm_89）全都没有，PTX 也没有 `atom.max.f32`（重载报错原文见反例 2）——浮点 max/min 出错
比"原子性欠缺"更具迷惑性。③ `_block`/`_system` 是**后缀选择 scope**；`cuda::atomic_ref`
是**模板参数**——两种 API 的 scope 表达位置不同，抄示例代码时别混。

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
- **不承诺「另一个 block 现在在跑」**。原子/自旋不会让别的 block 或别的 kernel 变得可调度；
  跨 block 的生产者-消费者要么有驻留/顺序保证（如 cooperative launch 的 `grid.sync()`），
  要么必须给自旋设界（见反例 1）。
- `_system` 原子在 system scope 下的原子性还有**附加条件**（PTX ISA §8.1.1
  *Limitations on atomicity at system scope*）——本机单卡不做实测。
- **默认 intrinsic 没有 order 参数**；想写「语义正确」的原子要用 libcu++ 的 `cuda::atomic` /
  `cuda::atomic_ref`（可指定 scope + order），它会被编译成 `atom.<order>.<scope>` / `ld.acquire` /
  `st.release` / `fence`。

**三个正交维度**（记住这张表就不会混）：

| 维度 | 取值 | 默认 | 表达方式 |
|---|---|---|---|
| 原子性 | 是/否 | — | `atomicAdd` vs `x = x + 1` |
| 作用域 scope | cta / gpu / sys | gpu(device) | `_block` / 无后缀 / `_system` |
| 顺序 order | relaxed / acquire / release / acq_rel / seq_cst | **relaxed** | `__threadfence*()`、`cuda::atomic_ref` |

### 三个轴真的独立吗——拿同一个操作转一圈

固定"操作 = `fetch_add(1)`",只动另外两个轴,行为差异全在 order/scope 上,操作本身的
"读旧值-加-写回-返回旧值"一字不变:

- `fetch_add(1, relaxed)` × device:不可分割地加;和谁都不排座次。
- `fetch_add(1, release)` × device:不可分割地加;且把「我之前的读写」排到它前面。本机
  SASS 落点是 `MEMBAR.ALL.GPU` + 原子指令(见 04「观察 3.5」)。
- `fetch_add(1, relaxed)` × cta(scope `_block`):只在 block 内不可分割不排座次;跨 block 的
  观察者不在 `_block` scope 的承诺范围里。

即:**relaxed 砍的是"排序",不是"原子性"**——原子 RMW 对原子对象的不可分割在任何 order 下都保留。
按同样方式,一个"读-改-写"给的参数空间是 5 种 order × 3 种 scope × 原子对象本身;04 的
order×scope 矩阵(`EXP-0004`)逐个量过本机落点,可交叉查阅。

### `fetch_add` 到底返回什么——old = 10, new = 11

`fetch_add` 容易被误读成"另一种内存顺序"或"另一种同步机制"。它不是——它就是 `atomicAdd`
的 libcu++ 对应物:**原子地把"旧值+增量"写回去,并返回旧值**(以及按指定的 order 携带的
任何排序效果):

```cpp
cuda::atomic_ref<int, cuda::thread_scope_device> c(x);
// 假设 x 当前值是 10。
// 两个线程各跑一行 c.fetch_add(1, …):
//   线程 A:读到旧值 10 → 写回 11 → 返回 10
//   线程 B:原子单元上只能串行——读到旧值 11 → 写回 12 → 返回 11
// 两行之后 x == 12;谁返回 10、谁返回 11 由硬件原子单元一次性裁决,两个 RMW 绝不撞车。
```

对照老接口:`atomicAdd(&x, 1)` 与 `c.fetch_add(1, relaxed)`**(同 scope=device)** 返回值
一样、原子性一样,唯一区别是后者多了 order 参数可传。`old=10, new=11` 这种对子是理解
所有 RMW 的钥匙:**返回的都是"动手之前"的值,新值只能再读回来或自己算**(新值 = 旧值 + 增量)。

## 反例

### 反例 1（核心，语义级）：「用了 atomic 就建立了同步/顺序」

> 验证维度：**排序 / 同步**（即便原子性完好，也无顺序、无可视性、无进度保证）——本反例同时踩了
> relaxed 原子不排序（前半）与跨 block 自旋进度不受控（后半「更要命的坑」）。

这是最贵的误解。经典 flag（旗子）+ payload（数据）生产者-消费者：

```cpp
// producer：先写 payload（普通 store），再用 relaxed 原子立旗
for (int i = 0; i < n; ++i) data[i] = i + 1;
atomicExch(&ready, 1);          // 默认 = relaxed + device，不给上面那批 store 排序

// consumer：relaxed 自旋等旗，看到旗后再读 payload
int s = 0;
while (atomicAdd(&ready, 0) == 0) {          // 有界自旋（真实代码见下方）
  if (++s >= kSpinLimit) break;              // kSpinLimit = 2,000,000
}
if (atomicAdd(&ready, 0) == 0) { *bad = -1; return; }   // 超时：报告 -1，不挂住
int b = 0;
for (int i = 0; i < n; ++i) if (data[i] != i + 1) ++b;
*bad = b;
```

上面是为了讲清楚而写的精简版；真实代码是 `experiments/EXP-0005-atomics/scripts/atomics_lab.cu` 里的
`pub_producer` / `pub_consumer`（一个 block、一个线程干活，`kSpinLimit` 同名）。

默认 `atomicExch` 是 relaxed：它**不**保证 producer 的 `data[]` 对 consumer 可见，也**不**保证
consumer 的 `data[]` 读发生在看到 `ready` 之后。这是一个**数据竞争（UB）**。

**修正一个容易背错的公式：「producer 的 fence = release、consumer 的 fence = acquire」不能当
机械等式用。** fence 不是贴在某行代码上的属性，它排的从来是「fence 前后的那批操作」——顺序由
一套完整的**通信协议**给出：producer 侧「普通 store 排班 → `__threadfence()` 掐住 → 原子立旗」，
consumer 侧对称地「原子读旗 → fence 掐住 → 普通 read」。三件缺一都拼不上：只 fence 不原子旗，
旗本身有竞争（丢更新，见反例 0）；只原子旗不 fence，`data[]` 根本没被排进旗的「之前」；两个
原语的作用域也要能对上（至少双方都有 device 及以上）。指南重构版 §5.7.5 给的就是这个
message-passing 形状。等价性来自「fence + 原子旗」的**组合**，不来自名字像。

**现代写法：把顺序直接写进原子本身（更难拼错）**——

```cpp
// scope 取 device（与默认 intrinsic 等价），order 显式给出
cuda::atomic_ref<int, cuda::thread_scope_device> ready_ref(ready);
// producer：写完 data[] 后立旗（release）——本机 SASS：MEMBAR.ALL.GPU + 原子/ST.E
ready_ref.store(1, cuda::memory_order_release);
// consumer：acquire 读旗，看到 1 才读 data[]——本机 SASS：LD.E + CCTL.IVALL（不是栅栏）
while (ready_ref.load(cuda::memory_order_acquire) == 0) { /* 有界，见下 */ }
```

fence 版本与这一版本在本机已核对到 SASS 层（**04 篇「观察 3.5」的 order×scope 矩阵**）：
release-acquire 的落点与「fence 三件套」等价、且不用绕常规栅栏。写新代码优先这种写法；
维护老代码时也按「排班-掐住-立旗」整套理解，别逐句背公式。

**更要命的坑：跨 block（或跨 grid）自旋等旗，可能永远等不到。**
CUDA **不保证**不同 block 同时驻留，更不保证两个 grid 同时在跑（见域 2）。设想 consumer 先被
调度上 SM、一直空转等旗，而 producer 因为没有空位始终启动不了——consumer 等的旗永远不会被立起，
它就**永不结束**（「不收敛」），而 producer 又在等 consumer 腾出位置，两边都卡住，这就是「死锁」。
本实验的 producer 与 consumer 恰好是**两个不同的 kernel**、分别提交到两条 stream，正是这种形状。
所以：

- **别写无界的 `while (atomicAdd(&ready,0)==0) {}`**。要么靠驻留/顺序保证（例如
  cooperative launch 的 `grid.sync()`，或保证两个 block 一定同时在跑），要么像上面一样
  给自旋加上界与超时。
- **有界自旋保证「一定会结束」**：至多自旋 `kSpinLimit` 次（外加上下文里的终检读），到点就放弃并把
  `bad` 置为 `-1`。这样不会死锁、不会挂住 GPU 和 host 的 `cudaDeviceSynchronize`。
- **有界超时只是教学保护，不是安全保证**：超时只说明「这次没等到」，它既不证明算法对，
  也不证明算法错；唯一作用是让程序**不会挂住**，并把「没等到」如实报出来。

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

> 验证维度：**表示与比较**（位模式序 ≠ 浮点数值序——找 max 的逻辑写错了，不是原子没起作用；
> 可与反例 0/1 对照：那次是"原子性缺失"，这次原子性完好，错的是拿什么当"大小"比）。

流行技巧是「浮点没有 `atomicMax`，那就把 `float` 重新解释成 `int` 再 `atomicMax`」。它对
**非负** 浮点碰巧成立（IEEE-754 正数的位模式随数值单调），但**两个负数就错**：

```cpp
atomicMax((int*)out, __float_as_int(v));   // 想当然
```

- `-5.0f = 0xC0A00000`，`-3.0f = 0xC0400000`。把这两个位模式**当成有符号 int** 来读：
  `0xC0A00000 = -1063256064`，`0xC0400000 = -1069547520`，所以
  **`0xC0A00000 > 0xC0400000`**（即 `-1063256064 > -1069547520`）。`atomicMax` 挑的是
  **位模式最大**的那个，于是选中 `0xC0A00000`，其浮点含义正是较小的 **-5.0f**。
- 两个负数为什么会反过来：负浮点的位模式最高位（符号位）恒为 1，剩下的位越大代表**绝对值越大、
  数值越小**。于是「位模式大」在正数区等于「数值大」，在负数区却等于「数值小」——同一个
  `atomicMax` 在两个区间里挑了相反的方向。（也可直接看成：负浮点的位模式随数值**单调递减**。）
- 本机实测：`max(-5.0, -3.0)` → naive 给 **-5.0（WRONG）**；正确写法（`atomicCAS` 循环 +
  `fmaxf`）给 **-3.0**。
- 更阴的是：两个都非负时没问题，一正一负也没问题（正浮点的位模式按有符号 int 读 ≥ 0、负浮点
  都 < 0，所以正的一定赢），**只有「两个负数」错**——只测几个用例很容易漏。

（`atomicMax` 的头文件重载只有整型：`int` / `unsigned` / `long long` / `unsigned long long`，
没有 `float`；PTX 也没有 `atom.max.f32`——本机 `ptxas -arch=sm_89` 对它报错：`Operation .max
requires .u32 or .s32 or .u64 or .s64 or .f16 or .f16x2 or .bf16 or .bf16x2 type`。所以
「浮点原子 max」本来就得自己用 `atomicCAS` 构造。`atomicCAS` 是**通用原子**，可自建任意 RMW，代价是在冲突下自旋重试。）

### atomicCAS 循环逐步读

反例 2 里那句"正确写法"的 `atomicCAS` 循环值得单独拆开——它是本篇唯一的**通用自建 RMW**：

```
        ┌─ ① 读旧值 ── assumed = *p（「动手前，我以为它还是它」）
        │
        ├─ ② 算候选 ── new = f(assumed)（这里 f = max(given, assumed)）
        │
        ├─ ③ 提交 ── old = atomicCAS(p, assumed, new)
        │           语义：*p 仍是 assumed ⇒ 写入 new（成功）
        │                 *p 已被别人换了 ⇒ 什么都不写（失败）
        │                 返回值永远是「提交瞬间 *p 的值」
        │
        └─ ④ 裁决 ── old == assumed ⇒ 成功，退出
                    old != assumed ⇒ 失败；返回值恰好是「插队者写下的值」
                    ⇒ 让 assumed ← old（直接拿到最新情报，少一次重读）→ 回 ②
```

几个非显然的要点：

- **CUDA 的 `atomicCAS` 不区分成败**：一律返回「提交瞬间 `*p` 的旧值」，成功与否要自己拿它和
  `assumed` 比（成功 ⇔ 没人插队 ⇔ 旧值还是期望值）。libcu++ 的
  `cuda::atomic_ref::compare_exchange_strong(expected, desired, …)` 才是"返回 bool、失败时
  自动把 `expected` 更新为当前值"的接口——就是上面 ④ 的自动化。
- **何时收敛**：对"候选只朝一个方向走"的 RMW（max 只增、计数只加、min 只减），每次成功提交都把
  `*p` 推离可反复抢的状态，取值范围有限，循环必然终结。**通用 RMW 没有单调性**，极端冲突下某线程
  可能长期反复插队（活锁）——这是所有 CAS 循环的通用风险，不是本例特有。
- **ABA 风险**：CAS 只比对"值"，不比对"历史"。若 `*p` 从 A 被改成 B 又改回 A,循环照样当
  "没变"提交成功。本篇的 max/计数类例子都是值单调的整数/位模式,ABA 无害;但在指针、双步更新、
  带版本号的场景（域 11/12 的对象）里它是经典陷阱,CUDA 没有 LL/SC,只能靠补版本号或换结构规避。
- **失败≠空闲**:CAS 失败的线程通常立刻重试,极端竞争下和自旋一样烧 SM 时间——这是
  "shared 原子便宜、global 原子贵"之外的第二笔账:**竞争强度**,不只是落点。
  （属成本直觉,不做定量基准;计数类证据见观察 4。）


### 反例 0（底座）：「`x = x + 1` 看起来像一条原子写」

> 验证维度：**原子性**（RMW 完整性）——最基本的"这一步不能被拆"。

源码一行，汇编是 load + add + store 三步。本机 256 线程 ×2000 次：预期 512000，非原子版只剩
**2000**（丢 510000）。原子性不是「编译器会自动帮我做对」。

## 本机验证（命令 + 观察）

所有产物在 `tmp/outputs/D05/`（未入库）。2026-10-11 用本机 scratch 目录（与前述工具链同参数）独立重编复验了一遍，未动 `tmp/`；本节表格与指令级引文均已被那次复验逐行核对。源码在 `experiments/EXP-0005-atomics/scripts/`：
`atomics_lab.cu`（功能）、`atomics_sass.cu`（指令）、`atomics_b128_sm90.cu`（边界）、
`sanitizer_smoke.cu`（工具自检）、`run.ps1`（一键跑）。

### 复现步骤

```powershell
$vcvars = 'E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$nvcc   = 'E:\NVIDIA\CUDA\v12.8\bin\nvcc.exe'
$cuobj  = 'E:\NVIDIA\CUDA\v12.8\bin\cuobjdump.exe'
$src    = 'E:\Project\cuda-semantics\experiments\EXP-0005-atomics\scripts'   # 源码（入库）
$out    = 'E:\Project\cuda-semantics\tmp\outputs\D05'                        # 产物（未入库）

# 功能实验
cmd /c "call `"$vcvars`" && `"$nvcc`" -O3 -lineinfo -arch=sm_89 -o `"$out\atomics_lab.exe`" `"$src\atomics_lab.cu`""
& "$out\atomics_lab.exe" atomicity
& "$out\atomics_lab.exe" floatmax
& "$out\atomics_lab.exe" publish 0 2000     # 无反例：0 bad；加 fence 也是 0；若超时会额外打印 timeouts=…

# 指令级：PTX + SASS
cmd /c "call `"$vcvars`" && `"$nvcc`" -arch=sm_89 -ptx   -o `"$out\atomics_sass.ptx`"   `"$src\atomics_sass.cu`""
cmd /c "call `"$vcvars`" && `"$nvcc`" -arch=sm_89 -cubin -o `"$out\atomics_sass.cubin`" `"$src\atomics_sass.cu`""
& $cuobj -sass "$out\atomics_sass.cubin" > "$out\atomics_sass.sass"
& $cuobj -sass "$out\atomics_lab.exe"    > "$out\atomics_lab.sass"   # 真实内核（publish）的 SASS

# 边界：128-bit 原子在 sm_89 应当编译失败，sm_90 通过
cmd /c "call `"$vcvars`" && `"$nvcc`" -arch=sm_89 -cubin -o `"$out\atomics_b128_sm89.cubin`" `"$src\atomics_b128_sm90.cu`""
cmd /c "call `"$vcvars`" && `"$nvcc`" -arch=sm_90 -cubin -o `"$out\atomics_b128_sm90.cubin`" `"$src\atomics_b128_sm90.cu`""
```

### 观察 1 — 设备/能力

`run_props.txt`：`RTX 4060 Laptop GPU`，compute capability **8.9**，warpSize 32，
SM 数 24，L2 **32 MiB**。

> **本节通用标签（适用于观察 2/3 的全部表格）**：
> ① 均为**本机 sm_89 观察**（编译参数见 §复现步骤），是"这台机器这份编译器是这样"的**实现证据**，
>   不是跨架构承诺；换 arch/flag/编译器版本都可能变（域 07 的"怎么读反汇编"就是为此）。
> ② **SASS 里的 `STRONG` 字样不要拿来反推 CUDA memory order**：本机 relaxed 与 acquire 的原子
>   指令同带 `.STRONG`（后者的排序来自随行的 `MEMBAR`/`CCTL.IVALL`，证据在 04 篇「观察 3.5」）。
>   SASS 助记符是硬件层的属性标签，与语言层 order 名字并无一一对应。
> ③ 表格只枚举函数级可复现的落点；PTX 还会随优化级别改写（域 07）。

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

> racecheck 旁证（2026-10-12，解锁后）：`race-naive` 报 hazard（点名 `atomics_lab.cu:43` 的
> 读写对），`race-atomic` 0 hazard——见观察 5。
| 浮点 max `floatmax` | -3.0 | naive **-5.0（WRONG）**；CAS+fmaxf -3.0 |
| 发布 `publish 0 200` / `publish 1 200` / `publish 0 2000` | 0 | **badRounds=0**（三组全部）；无反例——静默 UB；自旋有界，若等不到会额外打印 `timeouts=…` |
| 边界 `b128_sm89` | 编译失败 | `error: no instance of overloaded function "atomicAdd" matches … (float4 *, float4)` |
| 边界 `b128_sm90` | 通过 | 生成 `atomics_b128_sm90.cubin` |
| 超时路径（只跑 consumer，无 producer，有界） | 返回 `bad=-1` | **3/3 轮 `bad=-1`、无错误**，耗时约 **220–280 ms**（见下） |

**超时机制的实测（观察 4b）**：为了亲眼看到「等不到就放弃」确实会返回，用了 **只启动 consumer、
不启动 producer** 的探针（scratch 文件，不入库；按 `atomics_lab.cu` 里 `pub_consumer` 原样复刻），
两次独立各跑 3 轮：每轮 `err=no error bad=-1`，耗时 281.3 / 219.8 / 219.8 ms（首次）与
265.2 / 219.8 / 219.8 ms（2026-10-11 复验）。这说明：有界自旋在 `kSpinLimit = 2,000,000` 次读之后
确实结束，并把 `-1` 写回，kernel 正常返回，没有挂住。代价是：每次超时要白白烧掉约 0.2–0.3 s 的
SM 时间。（`atomics_lab.cu` 注释里写的 「~0.1-1 s」 与之相符。）

**与 `publish` 的关系**：正常运行时两个 kernel 都能驻留，producer 很快立旗，`publish` 200/2000 轮
全程未超时（外层计时的整个进程耗时：2026-10-11 复验 `publish 0 200` 约 1.28 s（含首次 CUDA 初始化）、
`publish 1 200` 约 0.10–0.13 s、`publish 0 2000` 约 0.21–0.23 s；若某个新进程先付冷初始化，
`publish 1 200` 也会显示 ~1.2 s。）也就是说，超时分支在本机**只在没有 producer 时**才会走到，
本实验**从未主动构造**「consumer 占满 SM、producer 启不来」的真死锁场景（那会永远等，故不做）。

### 观察 5 — compute-sanitizer（2026-10-12 更新：已解锁，本域反例拿到工具证据）

> **2026-10-11/12 更新**：此前长期"注入失败"的根因已查明——**Windhawk**（进程级 mod 注入平台）
> 与 sanitizer 的注入在 ntdll 撞车；停 Windhawk 服务后全部工具正常（细节与排查路径见 00 篇
> 「不承诺」与过程记忆 `tmp/docs/memory/windows-toolchain-pitfalls.md`）。**重启后自启**，
> 跑 sanitizer 前需再停——2026-10-12 的 racecheck 就是重新停过一次才跑成的。

解锁后本域补跑（产物 `tmp/outputs/sanitizer-2026-10-12/`）：

```
compute-sanitizer --tool racecheck atomics_lab.exe race-naive
  → Race reported between Read access at shared_counter_naive+0x80 (atomics_lab.cu:43)
    ... 1 hazard displayed
compute-sanitizer --tool racecheck atomics_lab.exe race-atomic
  → RACECHECK SUMMARY: 0 hazards
```

**对照双方一边一条、一边干净**：`race-naive` 的丢失更新正是把非原子 RMW 拆成
LDS/STS 呈现的读写对（工具直接点名 `atomics_lab.cu:43`），换成 `atomicAdd` 后同形状
代码 0 hazard。反例 0 的功能数字（256 vs 8192）之外，第一次有了**独立于本机数值的工具判定**。
同时这也校准了 racecheck 的适用范围：它看 **shared memory hazard**——本篇的
`publish`（global 通信）不在它的视野里（04 篇已实测同一结论），那组反例的证据仍以
PTX/SASS + 规范为准。

（以下为**解锁前**的历史记录，保留作"工具装了 ≠ 能用"的现场存档：`sanitizer_smoke.cu` 在
Windhawk 未停时同样报

```
========= COMPUTE-SANITIZER
========= Error: process didn't terminate successfully
========= Error: Target application terminated before first instrumented API call
```

当时已排除路径/参数/注册表（`EnableInterface` 本机已是 `0x1`），怀疑方向含管理员与驱动，
事后证实全部误判——真凶是 Windhawk 注入。这张表留着，作为"换台机器先跑 smoke 再下结论"的提醒。）

### 观察 6 — `atomicAdd(float)` 的 `.FTZ`：与 `-ftz` 无关（原未决项实测，2026-10-12）

`ftz_probe.cu`（`__global__ void k_ftz(float*, float) { atomicAdd(out, v); }`）同一源码用
`-ftz=false` / `-ftz=true` 各编一次，SASS **完全相同**：

```
RED.E.ADD.F32.FTZ.RN.STRONG.GPU
```

PTX 两版同样是 `atom.global.add.f32`——**根本不带 ftz qualifier**。结论（本机 sm_89 观察）：
浮点原子路径的 FTZ 是**指令形态固定的**，不受编译开关控制；这与普通浮点乘加的
`FADD.F32.RN`（`-ftz` 控制是否变 `.FTZ`）不同。域 8 讲 FTZ 时要"分开输入/中间值/输出、
分开指令路径"的提醒，在这里有了一个具体落点：**本机观察范围内，原子加的浮点路径固定 FTZ**，即使全局关掉
FTZ 也一样。（顺带佐证：publish 实验的 `atomicAdd(double)` 是 `…F64.RN…`，
double 无 FTZ 一族问题。）

### 产物清单

| 文件 | 内容 |
|---|---|
| `run.ps1` | 一键构建 + 全模式 + PTX/SASS + 边界编译（源码在 `experiments/EXP-0005-atomics/scripts/`） |
| `atomics_lab.cu` / `atomics_sass.cu` / `atomics_b128_sm90.cu` / `sanitizer_smoke.cu` | 源码副本（正本在 `experiments/EXP-0005-atomics/scripts/`） |
| `run_props.txt` / `run_atomicity.txt` / `run_floatmax.txt` / `run_publish_0_*.txt` / `run_race-*.txt` | 功能输出 |
| `atomics_sass.ptx` / `atomics_sass.sass` / `atomics_lab.sass` | 指令级证据 |
| `b128_sm89_build.log` / `b128_sm90_build.log` | 边界编译日志 |
| `racecheck_*.txt` | 解锁前的 sanitizer 失败记录（Windhawk 注入冲突，见观察 5） |

> 解锁后的补充产物不在本目录：`tmp/outputs/sanitizer-2026-10-12/d05-*`（race-naive 的
> racecheck 报告、race-atomic 干净对照、FTZ 双开关联测的 cubin/SASS/PTX）。
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

- **（已解决，2026-10-11/12）compute-sanitizer 本机不可用**：根因是 Windhawk 进程注入冲突（见
  观察 5 更新）；本域已拿到 shared memory 竞争的 racecheck 证据。**遗留**：`publish` 是 global
  广播竞争，racecheck 天生不查——它的独立判定仍缺工具层证据，见下一条。
- **反例 1 未在本机复现**（0/2400）。要把它变成「跑得出错」的 litmus，需要跨 SM 的 store-buffer /
  message-passing 变体（域 4 的核心工具），本机尚未搭。当前判定依赖规范 + SASS/PTX。
- **`publish` 的跨 grid 驻留问题未实测**：消费者已改为**有界自旋 + 超时报告**，本机已验证超时分支
  会正常返回（观察 4b）；但「consumer 先占住 SM、producer 永远启动不了」这一**真死锁**场景本机尚未构造
  （历史上 2400 轮全部在超时前收到旗子，本轮 200+200+2000 轮也是）。要复现，需要人为制造驻留压力
  （占满 SM 后再启动生产者）或换 cooperative launch 对照；这类实验会永远等，必须在外部计时守护下做。
- **（已解决，2026-10-11，04 篇「观察 3.5」）`ATOM…STRONG.GPU` 里 `STRONG` 与 relaxed/acquire 的
  精确对应**：order×scope 矩阵已实测——本机 SASS 助记符的 `STRONG` 是硬件层属性、不与
  memory order 一一对应（relaxed 原子同样 `STRONG`）；语义落点看随行的 `MEMBAR`/`CCTL.IVALL`。
  本篇观察 2/3 的通用标签②即为此结论。
- **（已解决，2026-10-12，观察 6）`atomicAdd(float)` SASS 出现 `.FTZ`**：与 `-ftz` 开关无关的
  指令形态固定行为（PTX 层面即无 ftz qualifier）。仍是**本机 sm_89 观察**，其他架构未测。
- **128-bit 原子（`.b128`）在 sm_90 的行为**未实测——本机无 sm_90，只证到「sm_89 编译失败」。
- **浮点原子求和的可复现性**只在概念层（非结合），未做「同输入不同分块得不同低位」的最小实验
  （域 8 的位置）。
- `_system` 原子的 system-scope 原子性附加条件（PTX §8.1.1）本机单卡不做实测。
