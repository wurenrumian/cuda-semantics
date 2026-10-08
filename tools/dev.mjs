#!/usr/bin/env node
// 一条命令同时起：本机 runner(8787) + VitePress dev(5173)。
// 用法: npm run dev
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const root = dirname(dirname(fileURLToPath(import.meta.url))) // tools/ -> 仓库根
const node = process.execPath
const vitepress = join(root, 'node_modules', 'vitepress', 'bin', 'vitepress.js')

const children = []
let closing = false

function shutdown(code = 0) {
  if (closing) return
  closing = true
  for (const c of children) {
    if (c.exitCode === null && !c.killed) {
      try { c.kill() } catch {}
    }
  }
  process.exit(code)
}

function start(label, args) {
  const c = spawn(node, args, { cwd: root, stdio: 'inherit' })
  c.on('exit', (code, signal) => {
    if (!closing) console.log(`[dev] ${label} 退出 (code=${code} signal=${signal})`)
    shutdown(code ?? 0)
  })
  c.on('error', (err) => {
    console.error(`[dev] ${label} 启动失败: ${err.message}`)
    shutdown(1)
  })
  children.push(c)
}

process.on('SIGINT', () => shutdown(0))
process.on('SIGTERM', () => shutdown(0))

console.log('[dev] runner -> http://127.0.0.1:8787')
console.log('[dev] site   -> http://localhost:5173')
console.log('[dev] Ctrl+C 停止两者')

start('runner', [join(root, 'tools', 'runner.mjs')])
start('site', [vitepress, 'dev', 'docs'])
