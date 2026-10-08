# 实验（实机运行）

实验定义与最小脚本在仓库的 `experiments/` 下。本页可**在本机直接运行**它们并查看输出。

> 前置：先在另一个终端启动本机 runner —— `npm run runner`（仅监听 `127.0.0.1:8787`）。
> 只允许运行 `experiments/*/run.ps1` 白名单内的脚本；执行发生在**本机**。

<script setup>
import { ref, onMounted } from 'vue'

const API = 'http://127.0.0.1:8787'
const exps = ref([])
const out = ref('')
const running = ref(false)
const err = ref('')

onMounted(async () => {
  try {
    const r = await fetch(`${API}/api/experiments`)
    exps.value = await r.json()
  } catch (e) {
    err.value = 'runner 未启动：先在另一个终端跑 `npm run runner`。'
  }
})

async function run(id) {
  running.value = true
  out.value = ''
  err.value = ''
  try {
    const r = await fetch(`${API}/api/run?id=${encodeURIComponent(id)}`, { method: 'POST' })
    if (!r.ok) throw new Error(await r.text())
    const reader = r.body.getReader()
    const dec = new TextDecoder()
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      out.value += dec.decode(value, { stream: true })
    }
  } catch (e) {
    err.value = String(e)
  }
  running.value = false
}
</script>

<div v-if="err" style="padding:10px;border:1px solid #c00;border-radius:6px;color:#c00">
  {{ err }}
</div>

<div v-if="!err && exps.length === 0" style="opacity:.8">
  暂无实验。把 `tmp/experiments/` 里的实验毕业到仓库 `experiments/` 后，这里会自动列出。
</div>

<div v-for="e in exps" :key="e.id" style="display:flex;gap:8px;align-items:center;margin:8px 0">
  <button :disabled="running" @click="run(e.id)">运行 {{ e.id }}</button>
  <code>{{ e.name }}</code>
</div>

<pre v-if="out" style="max-height:460px;overflow:auto;padding:12px;border-radius:6px">{{ out }}</pre>
