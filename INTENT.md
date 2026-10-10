# INTENT — CUDA 语义系统学习

> 本文件是项目的**北极星**：定义要产出什么、按什么标准、怎么分工。
> 多 agent 协作时，每个 worker 先读本文件，再读自己的域任务；产出必须满足这里的结构约定。

## 一句话

把 CUDA 的各类**语义**（执行模型 / 内存模型 / 同步 / 原子 / 指令 / 浮点 / 异步 / 编译器 / 架构）逐个吃透到 **能讲清楚、能举反例、能在本机验证**，并沉淀成可复读的学习材料。

## 为什么 / 成功标准

目标不是"会用 API"，而是理解每个语义 **承诺了什么、不承诺什么、边界在哪**。

每个域 Done 需同时满足：

1. **能讲** — 用自己的话讲清这个语义的承诺与边界。
2. **有反例** — 至少一个"想当然会错"的反例。
3. **本机验证** — 可复现的最小 kernel / 命令 / 观察（PTX / SASS / sanitizer / profiler）。
4. **有出处** — 引用官方文档章节或 PTX ISA 规范。

## 范围（域）

| # | 域 | 难度 |
|---|---|---|
| 0 | 环境基线（sm_89 / CUDA 12.8 / 工具链） | — |
| 1 | SIMT 执行模型 | 中 |
| 2 | 线程层次与调度 | 低 |
| 3 | 内存空间 | 低 |
| 4 | 内存一致性模型 | **高** |
| 5 | 原子 | **高** |
| 6 | 同步原语 | 中 |
| 7 | 指令级 / PTX | 中 |
| 8 | 浮点与数值 | 中 |
| 9 | 异步与流水 | 中 |
| 10 | 宿主侧运行时 / 启动 | 低 |
| 11 | 编译器行为与 UB | 中 |
| 12 | 架构 / 微架构相关的语义 | 中 |

详见 `tmp/docs/plans/cuda-semantics-topic-map.md`。

## 非目标

- 不做性能调优手册；性能只作为语义的"观测窗口"。
- 不追求覆盖全部 API；优先"语义有坑"的点。
- 不在本机跑 sm_90 专属特性（TMA / cluster / 128-bit 原子），只做概念了解。

## 产出物与结构

**每个域一篇**：起草在 `tmp/docs/research/<NN>-<slug>.md`，**审阅通过后毕业到 `docs/research/<NN>-<slug>.md`**。

固定小节：

```
一句话  →  承诺与边界  →  反例  →  本机验证（命令 + 观察）  →  出处  →  未决
```

**配套实验**：起草在 `tmp/experiments/EXP-XXXX-<slug>/`，毕业到 `experiments/EXP-XXXX-<slug>/`（定义 + 最小可复现脚本）；原始产物统一进 `tmp/outputs/`（**不进仓库**）。

## 仓库结构（已定 · 2026-10-08）

```
INTENT.md                       # 北极星（进仓库）
docs/                           # VitePress 内容源（进仓库）
  .vitepress/config.mts         # 站点配置 + 分域侧边栏
  index.md                      # 首页
  research/<NN>-<slug>.md       # 12 篇研究稿（毕业产物）
  experiments/EXP-XXXX-*.md     # 实验页：定义 + 一键脚本 + 预期输出
  evidence/                     # 精选证据（SASS 片段、ncu 摘要、图）
experiments/EXP-XXXX-<slug>/    # 实验定义 + 最小脚本（进仓库）
tmp/                            # 过程记忆（gitignored：journal/STATE/outputs/scratch）
```

## 毕业规则（已定）

`tmp/` 起草 → 审阅 → 毕业到仓库。**过程留 `tmp/`，结论进仓库**：

| 毕业（进 git） | 留 tmp（gitignored） |
|---|---|
| `docs/research/` 12 篇 + 索引 | `journal/`、`STATE.md` |
| `experiments/` 定义 + 最小脚本 | 原始 `outputs/`（`*.log`、SASS、`.exe/.cubin/.lib`） |
| 精选证据（内嵌/附录） | 半成品、`scratch/` |

- 毕业时**按仓库格式重写**，不是复制 `tmp/` 的排版；`tmp/` 原件保留（标 deprecated 或留档）。
- 每次毕业在 `STATE.md` 记一行「已毕业：<tmp 路径> → <仓库路径>」。

## 站点与实机实验（已定 · VitePress · 本机部署）

- VitePress 以 **`docs/` 为内容源**；研究稿即页面，按 12 个域做侧边栏。
- **本机部署**，不发布公网、不依赖 GitHub Pages：
  - 浏览：`npm run docs:dev`（localhost，热更新）
  - 构建 + 本机预览：`npm run docs:build` → `npm run docs:preview`
  - `base` 用 `/`（本机根路径服务即可）。
- 因为就在本机，"实机实验"可就地呈现：① 每篇「本机验证」给**可直接跑的命令 + 观察**；② `docs/experiments/` 成页，附脚本与预期输出；③ 精选产物（SASS/ncu 摘要/截图）作为页面附录。
- 执行环境 = **本机**（Windows + RTX 4060 / sm_89）。

### 本机 runner（已定）

- 一个**仅监听 `127.0.0.1`** 的本地服务（纯 Node，无框架依赖），实现"点按钮 → 本机跑 → 回显"。
- 接口：
  - `GET  /api/experiments` — 列出 `experiments/*/` 中有 `run.ps1` 的实验。
  - `POST /api/run?id=EXP-XXXX` — 在仓库根执行该实验的 `run.ps1`，stdout/stderr **流式**回传（chunked）。
- 安全边界（因为是本机执行）：**只允许白名单** —— `experiments/` 下、确有 `run.ps1` 的实验；不接受任意路径/命令；仅 `127.0.0.1`；校验 `Origin`；端口从候选列表自动选（`8787` 起，避开 Windows 保留端口段导致的 `EACCES`），可用 `RUNNER_PORT` 强制指定。
- 前端：VitePress 页面内联 Vue（`<script setup>`）**探测同一组候选端口**后调用 `http://127.0.0.1:<实际端口>/api/...`，展示实时输出。
- 启动：**只用 npm scripts** —— `npm run runner`（起 runner）、`npm run docs:dev`（起站点）。
- 访问：**仅本机 `localhost`**。

## 版本管理（已定）

- 先只做**本地 commit**（仓库当前零提交）；远端稍后再议。

## 环境约束（硬）

- GPU **sm_89（Ada, RTX 4060 Laptop）**，CUDA **12.8**。
- 编译：先 `call E:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`，再 `nvcc -arch=sm_89`。
- `compute-sanitizer` 2025.1.0 可用（racecheck / synccheck / initcheck）；profiler 用 `ncu` / `nsys`。
- 每个语义点尽量配一个可跑的验证，产物写 `tmp/outputs/`。

## 多 agent 分工（Orca 管理）

- 一个 **worker worktree** 认领一个域（域太大就拆成 2–3 篇 → 多个 worker）。
- 输入：本 `INTENT.md` + 该域在 topic map 里的条目。
- 输出：符合上面结构的 `tmp/docs/research/` 一篇 + 必要的实验证据。
- 主 worktree 负责**审阅 / 合并 / 毕业**到 `docs/`。
- 建议并行批次：
  - 批 1（基础、互相独立）：域 1、2、3、7、8。
  - 批 2（核心难点，依赖批 1 的直觉）：域 4、5、6。
  - 批 3：域 9、10、11、12。
- 依赖：域 4/5 最难且与域 6 强相关，放同一批或相邻批。
