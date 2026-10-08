# 实验（实机运行）

每个语义域配一个最小实验，源码在仓库 `experiments/EXP-00NN-*/` 下。
**结论与证据已整理在对应的研究稿里**；本页只用于「想重跑一遍」时点一下，输出会显示在下方。

> 前置：本机 runner 需在运行（`npm run dev` 已同时起好；或单独 `npm run runner`）。
> 仅执行白名单内的 `experiments/*/scripts/run.ps1`，**在本机**运行；产物写入 `tmp/outputs/`（不入库）。

<script setup>
import { ref, computed, onMounted } from 'vue'

const API = 'http://127.0.0.1:8787'
const exps = ref([])
const out = ref('')
const err = ref('')
const runningId = ref('')
const running = computed(() => runningId.value !== '')

const CN = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二']
function num(e) { return Number(e.id.slice(4)) }
function slug(e) { return e.name.replace(/^EXP-\d{4}-/, '') }
function docLink(e) { return `/research/${String(num(e)).padStart(2, '0')}-${slug(e)}` }
function label(e) { return `域 ${CN[num(e)] || num(e)} · ${slug(e)}` }

onMounted(async () => {
  try {
    const r = await fetch(`${API}/api/experiments`)
    if (!r.ok) throw new Error(await r.text())
    exps.value = await r.json()
  } catch (e) {
    err.value = '连不上本机 runner（127.0.0.1:8787）。请先运行 `npm run dev` 或 `npm run runner`，然后刷新本页。'
  }
})

async function run(e) {
  runningId.value = e.id
  out.value = ''
  err.value = ''
  try {
    const r = await fetch(`${API}/api/run?id=${encodeURIComponent(e.id)}`, { method: 'POST' })
    if (!r.ok) throw new Error(await r.text())
    const reader = r.body.getReader()
    const dec = new TextDecoder()
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      out.value += dec.decode(value, { stream: true })
    }
  } catch (e2) {
    err.value = String(e2)
  }
  runningId.value = ''
}
</script>

<div v-if="err" class="cs-alert">{{ err }}</div>

<div v-else-if="exps.length === 0" class="cs-muted">正在读取实验列表…</div>

<div class="cs-grid">
  <div v-for="e in exps" :key="e.id" class="cs-card">
    <div class="cs-card-head">
      <span class="cs-id">{{ e.id }}</span>
      <span class="cs-title">{{ label(e) }}</span>
    </div>
    <div class="cs-actions">
      <button class="cs-run" :disabled="running" @click="run(e)">
        {{ runningId === e.id ? '运行中…' : '▶ 运行' }}
      </button>
      <a class="cs-doc" :href="docLink(e)">看对应文档 →</a>
    </div>
  </div>
</div>

<div v-if="running || out" class="cs-console-wrap">
  <div class="cs-console-head">
    <span>{{ runningId ? runningId + ' 运行中…' : '输出' }}</span>
    <span class="cs-muted">结果与证据见对应研究稿；本页仅回显运行日志</span>
  </div>
  <pre class="cs-console">{{ out }}</pre>
</div>

<style scoped>
.cs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; margin: 20px 0; }
.cs-card { border: 1px solid var(--vp-c-divider); border-radius: 10px; padding: 14px 16px; background: var(--vp-c-bg-soft); }
.cs-card-head { display: flex; flex-direction: column; gap: 2px; margin-bottom: 12px; }
.cs-id { font-family: var(--vp-font-family-mono); font-size: 12px; color: var(--vp-c-brand-1); font-weight: 600; }
.cs-title { font-size: 15px; font-weight: 600; }
.cs-actions { display: flex; align-items: center; gap: 14px; }
.cs-run { background: var(--vp-c-brand-1); color: #fff; border: 0; border-radius: 8px; padding: 8px 16px; font-size: 14px; font-weight: 600; cursor: pointer; }
.cs-run:hover:not(:disabled) { background: var(--vp-c-brand-2); }
.cs-run:disabled { opacity: .5; cursor: default; }
.cs-doc { font-size: 13px; }
.cs-alert { padding: 12px 14px; border: 1px solid var(--vp-c-danger-1); border-radius: 8px; color: var(--vp-c-danger-1); margin: 16px 0; }
.cs-muted { opacity: .7; font-size: 13px; }
.cs-console-wrap { margin-top: 20px; border: 1px solid var(--vp-c-divider); border-radius: 10px; overflow: hidden; }
.cs-console-head { display: flex; justify-content: space-between; gap: 10px; padding: 8px 14px; background: var(--vp-c-bg-soft); font-size: 13px; }
.cs-console { margin: 0; max-height: 460px; overflow: auto; padding: 14px; font-size: 12.5px; line-height: 1.5; }
</style>
