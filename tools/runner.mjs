// Local experiment runner — available ONLY in the local-deployment scenario.
// Binds 127.0.0.1 and executes a whitelisted experiments/*/run.ps1, streaming
// stdout/stderr back to the page. Refuses anything that is not a real
// experiments/<name>/run.ps1 file. No arbitrary command execution.
import http from 'node:http'
import { spawn } from 'node:child_process'
import { readdirSync, existsSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const EXP_DIR = join(ROOT, 'experiments')
const HOST = '127.0.0.1'

// Windows 会给 Hyper-V/WSL 预留一批动态端口范围，落在其中的端口 listen 会报
// EACCES（不是 EADDRINUSE，netstat 也看不到占用）。所以不写死单一端口：
// RUNNER_PORT 指定则只用它；否则按顺序挑第一个能绑上的。前端会探测同一组端口。
const PORT_CANDIDATES = process.env.RUNNER_PORT
  ? [Number(process.env.RUNNER_PORT)]
  : [8787, 8123, 17387, 28123]
let PORT = PORT_CANDIDATES[0]

/** Only allow localhost origins (the VitePress dev/preview server). */
function corsFor(origin) {
  if (!origin) return ''
  try {
    const u = new URL(origin)
    if (u.hostname === 'localhost' || u.hostname === '127.0.0.1') return origin
  } catch {}
  return ''
}

/** Enumerate whitelisted experiments: experiments/<dir>/run.ps1 present. */
function listExperiments() {
  const out = []
  if (!existsSync(EXP_DIR)) return out
  for (const name of readdirSync(EXP_DIR)) {
    const dir = join(EXP_DIR, name)
    if (!statSync(dir).isDirectory()) continue
    let script = null
    for (const rel of ['run.ps1', 'scripts/run.ps1']) {
      if (existsSync(join(dir, rel))) {
        script = `experiments/${name}/${rel}`
        break
      }
    }
    if (!script) continue
    const id = (name.match(/^(EXP-\d{4})/) || [null, name])[1]
    out.push({ id, name, script })
  }
  out.sort((a, b) => a.id.localeCompare(b.id))
  return out
}

const server = http.createServer((req, res) => {
  const headers = {
    'Access-Control-Allow-Origin': corsFor(req.headers.origin) || 'null',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'content-type',
    'Cache-Control': 'no-store'
  }
  if (req.method === 'OPTIONS') {
    res.writeHead(204, headers)
    return res.end()
  }

  const url = new URL(req.url, `http://${HOST}:${PORT}`)

  if (url.pathname === '/api/experiments') {
    res.writeHead(200, { ...headers, 'content-type': 'application/json; charset=utf-8' })
    return res.end(JSON.stringify(listExperiments()))
  }

  if (url.pathname === '/api/run' && req.method === 'POST') {
    const id = url.searchParams.get('id') || ''
    const exp = listExperiments().find((e) => e.id === id || e.name === id)
    if (!exp) {
      res.writeHead(404, headers)
      return res.end(`unknown experiment: ${id}`)
    }
    res.writeHead(200, {
      ...headers,
      'content-type': 'text/plain; charset=utf-8',
      'X-Accel-Buffering': 'no'
    })
    const child = spawn('pwsh', ['-NoProfile', '-File', exp.script], { cwd: ROOT })
    child.stdout.on('data', (d) => res.write(d))
    child.stderr.on('data', (d) => res.write(d))
    child.on('error', (e) => res.write(`\n[spawn error] ${e}\n`))
    child.on('close', (code) => {
      res.write(`\n[exit ${code}]\n`)
      res.end()
    })
    return
  }

  res.writeHead(404, headers)
  res.end('not found')
})

// 依次尝试候选端口。EADDRINUSE=被占用；EACCES=落在 Windows 保留端口段。
function listen(ports, i = 0) {
  const port = ports[i]
  const onListening = () => {
    server.removeListener('error', onError)
    PORT = port
    console.log(`runner listening on http://${HOST}:${port}  (experiments dir: ${EXP_DIR})`)
  }
  const onError = (err) => {
    server.removeListener('listening', onListening)
    if (err.code === 'EADDRINUSE' || err.code === 'EACCES') {
      const why = err.code === 'EACCES' ? '被系统保留（excluded port range）' : '已被占用'
      if (i + 1 < ports.length) {
        console.warn(`[runner] ${HOST}:${port} 绑不上（${why}），改试 ${ports[i + 1]}`)
        return listen(ports, i + 1)
      }
      console.error(`[runner] 候选端口都绑不上：${ports.join(', ')}。请设 RUNNER_PORT=<空闲端口> 再试。`)
    } else {
      console.error(`[runner] 监听失败：${err.message}`)
    }
    process.exit(1)
  }
  server.once('error', onError)
  server.once('listening', onListening)
  server.listen(port, HOST)
}

listen(PORT_CANDIDATES)
