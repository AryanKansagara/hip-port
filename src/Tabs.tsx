import { useEffect, useMemo, useState } from 'react'
import { SEVERITIES, type Severity } from './engine/hazards.ts'
import { compareSnapshots, coveragePct, diffLines, readiness, STATUS_LABEL, worklistCsv, type ScanSnapshot, type WorkItem } from './engine/scan.ts'
import { rocmRequirements } from './engine/pydeps.ts'
import { DiffView } from './CodeView.tsx'
import { resolver, saveScan, slug, type Tab, type TabProps } from './Project.tsx'
import { getDefaults, type ItemStatus } from './store.ts'
import { Code, CopyButton, download, Empty, fmtPct, go, isTyping, Link, toast } from './ui.tsx'

export default function Tabs(props: TabProps & { tab: Tab }) {
  switch (props.tab) {
    case 'files': return <Files {...props} />
    case 'worklist': return <Worklist {...props} />
    case 'python': return <Python {...props} />
    case 'history': return <History {...props} />
    default: return <Overview {...props} />
  }
}

export const SEV_LABEL: Record<Severity, string> = { critical: 'Critical', high: 'High', medium: 'Medium', low: 'Low' }
const STATUS_TONE: Record<string, string> = { ok: 'good', partial: 'mid', deprecated: 'mid', unmapped: 'bad', unsupported: 'bad' }

function SevBar({ counts }: { counts: Record<Severity, number> }) {
  const total = SEVERITIES.reduce((s, k) => s + counts[k], 0)
  return (
    <div className="sevbar-wrap">
      <div className="sevbar" role="img" aria-label={SEVERITIES.map(s => `${counts[s]} ${s}`).join(', ')}>
        {total ? SEVERITIES.map(s => counts[s] ? <i key={s} className={s} style={{ flexGrow: counts[s] }} /> : null) : <i className="none" style={{ flexGrow: 1 }} />}
      </div>
      <div className="sev-legend">{SEVERITIES.map(s => <span key={s}><i className={`dot ${s}`} />{SEV_LABEL[s]} <b>{counts[s]}</b></span>)}</div>
    </div>
  )
}

function openCounts(items: WorkItem[], done: (k: string) => boolean) {
  return Object.fromEntries(SEVERITIES.map(s => [s, items.filter(i => i.severity === s && !done(i.key)).length])) as Record<Severity, number>
}

function Overview({ project, r }: TabProps) {
  const done = resolver(project)
  const counts = openCounts(r.items, done)
  const open = r.items.filter(i => !done(i.key))
  const cov = coveragePct(r.coverage)
  const last = project.snapshots.at(-1)
  const score = readiness(r.items, done)
  return (
    <div className="stack">
      <div className="kpis">
        <div className="kpi"><span>Readiness</span><b>{score}<small>/100</small></b><em className={last ? (score >= last.score ? 'good-text' : 'error') : 'muted'}>{last ? `${score - last.score >= 0 ? '+' : ''}${score - last.score} since “${last.label}”` : 'No saved scan yet'}</em></div>
        <div className="kpi"><span>API coverage</span><b>{fmtPct(cov)}</b><em className="muted">{r.coverage.converted} of {r.coverage.total} references mechanical</em></div>
        <div className="kpi"><span>Open items</span><b>{open.length}</b><em className="muted">{r.items.length - open.length} resolved</em></div>
        <div className="kpi"><span>Target</span><b className="mono">{r.target.id}</b><em className="muted">wave{r.target.wave} · {r.target.matrix}</em></div>
      </div>

      <div className="panel">
        <div className="row between wrap gap"><h2>Open work by severity</h2><Link href={`/app/p/${project.id}/worklist`} className="btn ghost sm">Worklist</Link></div>
        <SevBar counts={counts} />
        <ol className="top-items">
          {open.slice(0, 6).map(i => (
            <li key={i.key}>
              <span className={`sev ${i.severity}`}>{i.severity}</span>
              <div><b>{i.title}</b> <ItemLink project={project.id} i={i} /><p className="muted small">{i.detail}</p></div>
            </li>
          ))}
          {!open.length && <li className="muted">Nothing open. Save a scan to record it.</li>}
        </ol>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h2>Library mappings</h2>
          {r.libs.length ? (
            <div className="table-wrap"><table className="data">
              <thead><tr><th>CUDA</th><th>ROCm</th><th className="num">Refs</th><th>Status</th></tr></thead>
              <tbody>{r.libs.map(l => <tr key={l.lib}><td>{l.cuda}</td><td><b>{l.rocm}</b>{l.note && <div className="muted tiny">{l.note}</div>}</td><td className="num">{l.refs}</td><td><span className={`pill ${STATUS_TONE[l.worst]}`}>{l.worst === 'ok' ? 'mapped' : l.worst}</span></td></tr>)}</tbody>
            </table></div>
          ) : <p className="muted">No CUDA library calls found.</p>}
        </div>
        <div className="panel">
          <h2>ROCm readiness checklist</h2>
          <ul className="checklist">
            {r.checklist.map(c => (
              <li key={c.id} className={c.state}>
                <i aria-label={c.state === 'pass' ? 'Done' : c.state === 'todo' ? 'To do' : 'Note'} />
                <div><b>{c.title}</b><p className="muted small">{c.detail}</p>{c.command && <div className="cmd"><code>{c.command}</code><CopyButton text={c.command} /></div>}</div>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="panel">
        <h2>Files</h2>
        <FileTable project={project.id} r={r} />
      </div>
    </div>
  )
}

function FileTable({ project, r }: { project: string; r: TabProps['r'] }) {
  return (
    <div className="table-wrap"><table className="data">
      <thead><tr><th>File</th><th>Output</th><th className="num">Lines</th><th className="num">Coverage</th>{SEVERITIES.map(s => <th key={s} className="num">{SEV_LABEL[s]}</th>)}</tr></thead>
      <tbody>{r.files.map(f => (
        <tr key={f.path}>
          <td><Link href={`/app/p/${project}/files?f=${encodeURIComponent(f.path)}`} className="mono">{f.path}</Link></td>
          <td className="mono muted">{f.plain ? 'unchanged' : f.outPath}</td>
          <td className="num">{f.lines}</td>
          <td className="num">{f.coverage.total ? fmtPct(coveragePct(f.coverage)) : '—'}</td>
          {SEVERITIES.map(s => { const n = f.hazards.filter(h => h.severity === s).length; return <td key={s} className={`num${n ? ' sevnum ' + s : ' muted'}`}>{n || '·'}</td> })}
        </tr>
      ))}</tbody>
    </table></div>
  )
}

function ItemLink({ project, i }: { project: string; i: WorkItem }) {
  if (i.category === 'dependency') return <Link href={`/app/p/${project}/python`} className="mono small">{i.file}{i.line ? `:${i.line}` : ''}</Link>
  return <Link href={`/app/p/${project}/files?f=${encodeURIComponent(i.file)}&l=${i.line}`} className="mono small">{i.file}:{i.line}</Link>
}

// ---------- Files ----------

function Files({ project, r }: TabProps) {
  const q = new URLSearchParams(location.search)
  const sel = r.files.find(f => f.path === q.get('f')) ?? r.files.find(f => f.hazards.length) ?? r.files[0]
  const [line, setLine] = useState<number | undefined>(Number(q.get('l')) || undefined)
  const [wrap, setWrap] = useState(getDefaults().wrap)
  const rows = useMemo(() => sel ? diffLines(sel.source, sel.translated) : [], [sel])
  useEffect(() => {
    if (!line) return
    document.getElementById(`L${line}`)?.scrollIntoView({ block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' })
  }, [line, sel?.path])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || !sel || (e.key !== 'j' && e.key !== 'k')) return
      const ls = [...new Set(sel.hazards.map(h => h.line))]
      if (!ls.length) return
      const cur = line ?? 0
      const next = e.key === 'j' ? ls.find(l => l > cur) ?? ls[0] : [...ls].reverse().find(l => l < cur) ?? ls[ls.length - 1]
      setLine(next)
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  })
  if (!sel) return <Empty title="No source files"><Link href={`/app/p/${project.id}/sources`} className="btn primary">Add sources</Link></Empty>
  const pick = (path: string) => { setLine(undefined); go(`/app/p/${project.id}/files?f=${encodeURIComponent(path)}`, { replace: true, keepScroll: true }) }
  return (
    <div className="files-layout">
      <aside className="file-list" aria-label="Files">
        {r.files.map(f => {
          const crit = f.hazards.filter(h => h.severity === 'critical' || h.severity === 'high').length
          return (
            <button key={f.path} type="button" className={f.path === sel.path ? 'on' : ''} aria-current={f.path === sel.path} onClick={() => pick(f.path)}>
              <span className="mono">{f.path}</span>
              <span className="badges">{crit ? <b className="bad">{crit}</b> : null}{f.hazards.length - crit ? <b className="mid">{f.hazards.length - crit}</b> : null}{f.plain && <small className="muted">plain C++</small>}</span>
            </button>
          )
        })}
      </aside>
      <div className="file-main">
        <div className="row between wrap gap file-bar">
          <div className="mono small"><b>{sel.path}</b> <span className="muted">→ {sel.outPath}</span></div>
          <div className="row gap-sm wrap">
            <label className="check small"><input type="checkbox" checked={wrap} onChange={e => setWrap(e.target.checked)} /> Wrap</label>
            <CopyButton text={sel.translated} label="Copy HIP" />
            <button type="button" className="btn ghost sm" onClick={() => download(sel.outPath.split('/').pop()!, sel.translated, 'text/plain')}>Download file</button>
          </div>
        </div>
        <DiffView rows={rows} left={sel.source} right={sel.translated} hazards={sel.hazards} focusLine={line} onPick={setLine} wrap={wrap} />
        <div className="panel hz-panel">
          <h3>{sel.hazards.length ? `${sel.hazards.length} hazard${sel.hazards.length > 1 ? 's' : ''} in this file` : 'No hazards in this file'} <small className="muted">J / K to step</small></h3>
          {sel.hazards.map((h, i) => (
            <button key={i} type="button" className={`hz-item${h.line === line ? ' on' : ''}`} onClick={() => setLine(h.line)}>
              <span className={`sev ${h.severity}`}>{h.severity}</span>
              <span><b>Line {h.line}</b> <code>{h.snippet}</code><br />{h.message}<br /><em>Fix: {h.fix}</em></span>
            </button>
          ))}
          {sel.uses.filter(u => u.status !== 'ok').map(u => (
            <div key={u.kind + u.cuda} className="hz-item static">
              <span className={`pill ${STATUS_TONE[u.status]}`}>{u.status}</span>
              <span><code>{u.cuda}</code>{u.hip ? <> → <code>{u.hip}</code></> : null} <span className="muted">line {u.lines.join(', ')}</span>{u.note && <><br />{u.note}</>}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ---------- Worklist ----------

function Worklist({ project, r, update }: TabProps) {
  const [sev, setSev] = useState<Severity | 'all'>('all')
  const [cat, setCat] = useState<string>('all')
  const [show, setShow] = useState<'open' | 'all' | 'resolved'>('open')
  const [qText, setQ] = useState('')
  const status = project.status
  const list = r.items.filter(i =>
    (sev === 'all' || i.severity === sev) && (cat === 'all' || i.category === cat) &&
    (show === 'all' || (show === 'open' ? !status[i.key] : !!status[i.key])) &&
    (!qText || (i.title + i.file + i.detail).toLowerCase().includes(qText.toLowerCase())))
  const setStatus = (k: string, s: ItemStatus | 'open') => {
    const next = { ...status }
    if (s === 'open') delete next[k]; else next[k] = s
    update({ status: next })
  }
  const done = Object.keys(status).filter(k => r.items.some(i => i.key === k)).length
  return (
    <div className="stack">
      <div className="row between wrap gap">
        <div className="filters">
          <select value={sev} onChange={e => setSev(e.target.value as Severity | 'all')} aria-label="Severity"><option value="all">All severities</option>{SEVERITIES.map(s => <option key={s} value={s}>{SEV_LABEL[s]}</option>)}</select>
          <select value={cat} onChange={e => setCat(e.target.value)} aria-label="Category"><option value="all">All categories</option><option value="hazard">Hazards</option><option value="api">API mappings</option><option value="header">Headers</option><option value="dependency">Python deps</option></select>
          <select value={show} onChange={e => setShow(e.target.value as 'open')} aria-label="Status"><option value="open">Open</option><option value="resolved">Resolved</option><option value="all">All</option></select>
          <input type="search" placeholder="Filter" value={qText} onChange={e => setQ(e.target.value)} aria-label="Filter text" />
        </div>
        <div className="row gap-sm">
          <span className="muted small">{done} of {r.items.length} resolved</span>
          <button type="button" className="btn outline sm" onClick={() => download(`${slug(project.name)}-worklist.csv`, worklistCsv(r.items, k => status[k] ?? 'open'))}>Export CSV</button>
          <button type="button" className="btn ghost sm" onClick={() => saveScan(project, r, update)}>Save scan</button>
        </div>
      </div>
      <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={r.items.length} aria-valuenow={done} aria-label="Resolved items"><i style={{ width: `${r.items.length ? (done / r.items.length) * 100 : 100}%` }} /></div>
      {list.length ? (
        <div className="table-wrap"><table className="data worklist">
          <thead><tr><th>Severity</th><th>Item</th><th>Where</th><th>Status</th></tr></thead>
          <tbody>{list.map(i => (
            <tr key={i.key} className={status[i.key] ? 'resolved' : ''}>
              <td><span className={`sev ${i.severity}`}>{i.severity}</span></td>
              <td><b>{i.title}</b><div className="muted small">{i.detail}</div>{i.fix && <div className="small fix">{i.fix}</div>}</td>
              <td><ItemLink project={project.id} i={i} /></td>
              <td>
                <select value={status[i.key] ?? 'open'} onChange={e => setStatus(i.key, e.target.value as ItemStatus)} aria-label={`Status for ${i.title}`}>
                  <option value="open">Open</option><option value="done">Done</option><option value="wontfix">Won't fix</option>
                </select>
              </td>
            </tr>
          ))}</tbody>
        </table></div>
      ) : <Empty title={show === 'open' && !qText && sev === 'all' && cat === 'all' ? 'Nothing open' : 'No items match'}><p className="muted">{show === 'open' ? 'Every item is resolved or filtered out.' : 'Try another filter.'}</p></Empty>}
    </div>
  )
}

// ---------- Python ----------

const DEP_TONE: Record<string, string> = { ok: 'good', rocm: 'good', fork: 'mid', partial: 'mid', 'cuda-only': 'bad', unknown: 'none' }

function Python({ project, r }: TabProps) {
  if (!r.deps) return (
    <Empty title="No Python dependency file">
      <p className="muted">Add requirements.txt or environment.yml on the Sources tab to check torch, attention kernels, quantization libraries and CUDA-only packages.</p>
      <Link href={`/app/p/${project.id}/sources`} className="btn primary">Add a dependency file</Link>
    </Empty>
  )
  const d = r.deps
  const req = rocmRequirements(d)
  const by = (s: string) => d.deps.filter(x => x.status === s).length
  return (
    <div className="stack">
      <div className="kpis">
        <div className="kpi"><span>Packages</span><b>{d.deps.length}</b><em className="muted">{r.depsName}</em></div>
        <div className="kpi"><span>Ready on ROCm</span><b className="good-text">{by('ok') + by('rocm')}</b><em className="muted">pure Python or ROCm wheels</em></div>
        <div className="kpi"><span>Forks or partial</span><b className="warn-text">{by('fork') + by('partial')}</b><em className="muted">need a different source</em></div>
        <div className="kpi"><span>CUDA only</span><b className="error">{by('cuda-only')}</b><em className="muted">remove or replace</em></div>
      </div>
      {(d.cudaIndexes.length > 0 || d.cudaChannels.length > 0 || d.errors.length > 0) && (
        <div className="callout warn">
          {d.cudaIndexes.map(x => <p key={x.line}>Line {x.line}: package index <code>{x.url}</code> serves CUDA wheels.</p>)}
          {d.cudaChannels.map(c => <p key={c}>Conda channel <code>{c}</code> only serves CUDA packages.</p>)}
          {d.errors.map(e => <p key={e.line}>Line {e.line}: {e.message}</p>)}
        </div>
      )}
      <div className="table-wrap"><table className="data">
        <thead><tr><th>Package</th><th>Spec</th><th>Status</th><th>What to do</th><th>Checked</th></tr></thead>
        <tbody>{d.deps.map((x, i) => (
          <tr key={i}>
            <td className="nowrap"><b className="mono">{x.name}</b> <span className="muted tiny">line {x.line}{x.source === 'conda' ? ' · conda' : ''}</span></td>
            <td className="mono small">{x.spec || '—'}{x.cudaLocal && <span className="pill bad">+{x.cudaLocal}</span>}</td>
            <td><span className={`pill ${DEP_TONE[x.status]}`}>{STATUS_LABEL[x.status]}</span></td>
            <td className="small"><div>{x.action}</div><div className="muted">{x.note}</div></td>
            <td className="small muted">{x.checked ?? '—'}</td>
          </tr>
        ))}</tbody>
      </table></div>
      <Code code={req} label="requirements-rocm.txt" />
      <div className="row gap"><button type="button" className="btn outline" onClick={() => download('requirements-rocm.txt', req, 'text/plain')}>Download requirements-rocm.txt</button></div>
      <p className="muted small">Package statuses come from a curated table last checked in October 2026. ROCm moves quickly: confirm versions against the release notes for your ROCm version.</p>
    </div>
  )
}

// ---------- History ----------

function History({ project, r, update }: TabProps) {
  const snaps = project.snapshots
  const [a, setA] = useState(snaps.at(-2)?.id ?? '')
  const [b, setB] = useState(snaps.at(-1)?.id ?? '')
  const sa = snaps.find(s => s.id === a), sb = snaps.find(s => s.id === b)
  const cmp = sa && sb && sa.id !== sb.id ? compareSnapshots(sa, sb) : null
  if (!snaps.length) return (
    <Empty title="No scans saved yet">
      <p className="muted">Save a scan to start tracking progress. Each import saves one automatically.</p>
      <button type="button" className="btn primary" onClick={() => saveScan(project, r, update)}>Save this scan</button>
    </Empty>
  )
  const remove = (s: ScanSnapshot) => {
    const prev = project.snapshots
    update({ snapshots: prev.filter(x => x.id !== s.id) })
    toast(`Deleted “${s.label}”`, { undo: () => update({ snapshots: prev }) })
  }
  return (
    <div className="stack">
      <div className="panel">
        <div className="row between wrap gap"><h2>Progress</h2><button type="button" className="btn outline sm" onClick={() => saveScan(project, r, update)}>Save current scan</button></div>
        <ProgressChart snaps={snaps} />
      </div>
      <div className="panel">
        <h2>Compare scans</h2>
        <div className="row gap wrap">
          <label className="field inline"><span>From</span><select value={a} onChange={e => setA(e.target.value)}>{snaps.map(s => <option key={s.id} value={s.id}>{s.label} · {new Date(s.at).toLocaleDateString()}</option>)}</select></label>
          <label className="field inline"><span>To</span><select value={b} onChange={e => setB(e.target.value)}>{snaps.map(s => <option key={s.id} value={s.id}>{s.label} · {new Date(s.at).toLocaleDateString()}</option>)}</select></label>
        </div>
        {cmp ? (
          <div className="grid-2 cmp">
            <div><h3 className="good-text">{cmp.fixed.length} fixed</h3><ul className="cmp-list">{cmp.fixed.map(i => <li key={i.key}><span className={`sev ${i.severity}`}>{i.severity}</span> {i.title} <span className="muted mono tiny">{i.file}</span></li>)}</ul></div>
            <div><h3 className={cmp.added.length ? 'error' : 'muted'}>{cmp.added.length} new</h3><ul className="cmp-list">{cmp.added.map(i => <li key={i.key}><span className={`sev ${i.severity}`}>{i.severity}</span> {i.title} <span className="muted mono tiny">{i.file}</span></li>)}</ul>
              <p className="muted small">{cmp.still} still open · readiness {cmp.scoreDelta >= 0 ? '+' : ''}{cmp.scoreDelta}</p></div>
          </div>
        ) : <p className="muted">Pick two different scans.</p>}
      </div>
      <div className="table-wrap"><table className="data">
        <thead><tr><th>Scan</th><th>Date</th><th>Target</th><th className="num">Files</th><th className="num">Coverage</th>{SEVERITIES.map(s => <th key={s} className="num">{SEV_LABEL[s]}</th>)}<th className="num">Readiness</th><th /></tr></thead>
        <tbody>{[...snaps].reverse().map(s => (
          <tr key={s.id}>
            <td><b>{s.label}</b></td><td>{new Date(s.at).toLocaleString()}</td><td className="mono">{s.target}</td><td className="num">{s.files}</td><td className="num">{fmtPct(coveragePct(s.coverage))}</td>
            {SEVERITIES.map(k => <td key={k} className="num">{s.counts[k]}</td>)}
            <td className="num"><b>{s.score}</b></td>
            <td><button type="button" className="btn danger-ghost sm" onClick={() => remove(s)} aria-label={`Delete ${s.label}`}>Delete</button></td>
          </tr>
        ))}</tbody>
      </table></div>
    </div>
  )
}

function ProgressChart({ snaps }: { snaps: ScanSnapshot[] }) {
  const W = 1000, H = 260, P = { l: 20, r: 40, t: 24, b: 34 }
  const iw = W - P.l - P.r, ih = H - P.t - P.b
  const maxItems = Math.max(1, ...snaps.map(s => SEVERITIES.reduce((n, k) => n + s.counts[k], 0)))
  const bw = Math.min(46, iw / snaps.length / 1.8)
  const x = (i: number) => P.l + bw + (snaps.length === 1 ? (iw - 2 * bw) / 2 : (i * (iw - 2 * bw)) / (snaps.length - 1))
  const line = snaps.map((s, i) => `${i ? 'L' : 'M'}${x(i)},${P.t + ih - (s.score / 100) * ih}`).join('')
  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={`Readiness over ${snaps.length} scans: ${snaps.map(s => `${s.label} ${s.score}`).join(', ')}`}>
        {[0, 50, 100].map(v => <g key={v}><line x1={P.l} x2={W - P.r} y1={P.t + ih - (v / 100) * ih} y2={P.t + ih - (v / 100) * ih} className="grid" /><text x={W - P.r + 6} y={P.t + ih - (v / 100) * ih + 4} className="axis">{v}</text></g>)}
        {snaps.map((s, i) => {
          let y = P.t + ih
          return <g key={s.id}>{[...SEVERITIES].reverse().map(k => { const h = (s.counts[k] / maxItems) * ih * 0.9; y -= h; return h ? <rect key={k} x={x(i) - bw / 2} y={y} width={bw} height={h} className={`bar ${k}`}><title>{`${s.counts[k]} ${k}`}</title></rect> : null })}<text x={x(i)} y={H - 12} textAnchor="middle" className="axis">{s.label.length > 18 ? s.label.slice(0, 17) + '…' : s.label}</text></g>
        })}
        <path d={line} className="score-line" />
        {snaps.map((s, i) => <g key={s.id}><circle cx={x(i)} cy={P.t + ih - (s.score / 100) * ih} r="4.5" className="score-dot" /><text x={x(i)} y={P.t + ih - (s.score / 100) * ih - 10} textAnchor="middle" className="score-lbl">{s.score}</text></g>)}
      </svg>
      <div className="sev-legend"><span><i className="dot score" />Readiness (0–100)</span>{SEVERITIES.map(s => <span key={s}><i className={`dot ${s}`} />{SEV_LABEL[s]} open</span>)}</div>
    </div>
  )
}
