# 研究：环境基线 + CUDA 语义的 12 个域

> 每篇结构：**一句话 → 承诺与边界 → 反例 → 本机验证（命令 + 观察）→ 出处**
>
> 本机基线：RTX 4060 Laptop（sm_89 / Ada）· CUDA 12.8 · compute-sanitizer 2025.1.0

13 篇（环境基线 + 12 个语义域）均已毕业：先在 `tmp/docs/research/` 起草，按仓库格式重写后落进 `docs/research/`。

另加一篇跨域**速查**（13 · 原语速查）：把各域散落的原语汇到一处，每个给一句人话语义；定位是"认脸 + 指路"，不替代各域的承诺与边界。函数名从本机头文件批量抽出（CUDA 12.8 / sm_89）。

| # | 域 | 页面 |
|---|---|---|
| 00 | 环境基线（语义天花板） | [00-baseline](./00-baseline) |
| 01 | SIMT 执行模型 | [01-simt-execution-model](./01-simt-execution-model) |
| 02 | 线程层次与调度 | [02-thread-hierarchy](./02-thread-hierarchy) |
| 03 | 内存空间 | [03-memory-spaces](./03-memory-spaces) |
| 04 | 内存一致性模型 | [04-memory-consistency-model](./04-memory-consistency-model) |
| 05 | 原子 | [05-atomics](./05-atomics) |
| 06 | 同步原语 | [06-synchronization](./06-synchronization) |
| 07 | 指令级 / PTX | [07-ptx-instruction-level](./07-ptx-instruction-level) |
| 08 | 浮点与数值 | [08-floating-point](./08-floating-point) |
| 09 | 异步与流水 | [09-async-pipelines](./09-async-pipelines) |
| 10 | 宿主侧运行时 / 启动 | [10-host-runtime](./10-host-runtime) |
| 11 | 编译器行为与 UB | [11-compiler-and-ub](./11-compiler-and-ub) |
| 12 | 架构 / 微架构相关的语义 | [12-architecture-microarchitecture](./12-architecture-microarchitecture) |
| 13 | 原语速查（跨域 · 一句话语义） | [13-primitive-reference](./13-primitive-reference) |
