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
const PORT = Number(process.env.RUNNER_PORT || 8787)

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

server.listen(PORT, HOST, () => {
  console.log(`runner listening on http://${HOST}:${PORT}  (experiments dir: ${EXP_DIR})`)
})
