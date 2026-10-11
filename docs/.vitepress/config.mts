import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'CUDA 语义',
  description: '在本机（sm_89）验证的 CUDA 语义笔记',
  lang: 'zh-CN',
  base: '/',
  // 毕业前正文尚未全部落盘，先容忍死链；毕业完成后移除。
  ignoreDeadLinks: true,
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '研究', link: '/research/' },
      { text: '实验', link: '/experiments/' }
    ],
    sidebar: {
      '/research/': [
        {
          text: 'CUDA 语义（环境基线 + 12 域）',
          items: [
            { text: '00 · 环境基线', link: '/research/00-baseline' },
            { text: '01 · SIMT 执行模型', link: '/research/01-simt-execution-model' },
            { text: '02 · 线程层次与调度', link: '/research/02-thread-hierarchy' },
            { text: '03 · 内存空间', link: '/research/03-memory-spaces' },
            { text: '04 · 内存一致性模型', link: '/research/04-memory-consistency-model' },
            { text: '05 · 原子', link: '/research/05-atomics' },
            { text: '06 · 同步原语', link: '/research/06-synchronization' },
            { text: '07 · 指令级 / PTX', link: '/research/07-ptx-instruction-level' },
            { text: '08 · 浮点与数值', link: '/research/08-floating-point' },
            { text: '09 · 异步与流水', link: '/research/09-async-pipelines' },
            { text: '10 · 宿主侧运行时', link: '/research/10-host-runtime' },
            { text: '11 · 编译器与 UB', link: '/research/11-compiler-and-ub' },
            { text: '12 · 架构 / 微架构', link: '/research/12-architecture-microarchitecture' }
          ]
        }
      ]
    },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    search: { provider: 'local' }
  }
})
