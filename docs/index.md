---
layout: home

hero:
  name: CUDA 语义
  text: 在本机 sm_89 上验证的语义笔记
  tagline: 执行模型 · 内存模型 · 原子 · 同步 · PTX · 架构
  actions:
    - theme: brand
      text: 开始阅读
      link: /research/
    - theme: alt
      text: 实验
      link: /experiments/

features:
  - title: 环境基线 + 12 个语义域
    details: 先钉死本机基线（sm_89 的语义天花板），再从 SIMT 到内存一致性；每篇都含「承诺与边界 / 反例 / 本机可复现验证 / 出处 / 未决」。
  - title: 实机可跑
    details: 实验定义与最小脚本随仓库；本机 runner 让页面直接执行并把输出回显。
  - title: 本机部署
    details: VitePress 站点仅在 localhost 运行，不发布公网；执行环境就是这台 RTX 4060 / CUDA 12.8。
---
