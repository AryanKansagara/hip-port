import { lazy, Suspense, useEffect, useState } from 'react'
import { TARGETS } from './engine/gfx.ts'
import { coveragePct, readiness, snapshotOf, worklistCsv, type ScanResult } from './engine/scan.ts'
import { rocmRequirements } from './engine/pydeps.ts'
import { writeZip } from './engine/zip.ts'
import { useProject, useScan } from './projectData.ts'
import { uuid, type Dataset, type Project } from './store.ts'
import { download, Empty, fmtPct, go, isTyping, Link, Modal, SyntheticBadge, toast } from './ui.tsx'

const Tabs = lazy(() => import('./Tabs.tsx'))
const Sources = lazy(() => import('./Import.tsx'))

export const TABS = [['overview', 'Overview'], ['files', 'Files & diff'], ['worklist', 'Worklist'], ['python', 'Python stack'], ['history', 'History'], ['sources', 'Sources']] as const
export type Tab = (typeof TABS)[number][0]

export interface TabProps {
  project: Project
  ds: Dataset
  r: ScanResult
  update(p: Partial<Project>): void
}

export const resolver = (p: Project) => (k: string) => !!p.status[k]

export function saveScan(project: Project, r: ScanResult, update: (p: Partial<Project>) => void, label?: string) {
  const snap = snapshotOf(r, uuid(), label ?? `Scan ${project.snapshots.length + 1}`, Date.now(), resolver(project))
  update({ snapshots: [...project.snapshots, snap] })
  toast(`Saved “${snap.label}”: readiness ${snap.score}`, { tone: 'ok', undo: () => update({ snapshots: project.snapshots }) })
}

export function downloadZip(project: Project, r: ScanResult) {
  const files = r.files.map(f => ({ path: f.outPath, text: f.translated }))
  if (r.deps) files.push({ path: 'requirements-rocm.txt', text: rocmRequirements(r.deps) })
  files.push({ path: 'HIP_PORT_WORKLIST.csv', text: worklistCsv(r.items, k => project.status[k] ?? 'open') })
  download(`${slug(project.name)}-hip.zip`, new Blob([writeZip(files) as BlobPart], { type: 'application/zip' }))
}

export const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'project'

export default function ProjectView({ id, tab: rawTab }: { id: string; tab: string }) {
  const { loading, project, ds, update } = useProject(id)
  const r = useScan(project, ds)
  const tab = (TABS.some(t => t[0] === rawTab) ? rawTab : 'overview') as Tab
  const [help, setHelp] = useState(false)

  useEffect(() => { if (project) document.title = `${project.name} · HIP Port` }, [project?.name])

  const setTab = (t: Tab) => go(`/app/p/${id}${t === 'overview' ? '' : '/' + t}`, { keepScroll: true })

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey || !project) return
      const n = Number(e.key)
      if (n >= 1 && n <= TABS.length) { e.preventDefault(); setTab(TABS[n - 1][0]) }
      else if (e.key === '?') setHelp(true)
      else if (e.key === 's' && r) saveScan(project, r, update)
      else if (e.key === 'p' && r) go(`/app/p/${id}/report`)
      else if (e.key === 'd' && r) downloadZip(project, r)
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  })

  if (loading) return <div className="boot" role="status"><span className="spinner" /> Translating sources</div>
  if (!project || !ds) return (
    <section className="page narrow center">
      <Empty title="Project not found"><p className="muted">It may have been deleted, or it lives on another device.</p><Link href="/app" className="btn primary">All projects</Link></Empty>
    </section>
  )
  const empty = !r
  const score = r ? readiness(r.items, resolver(project)) : null

  return (
    <section className="page">
      <div className="proj-head">
        <div className="proj-title">
          <div className="eyebrow"><Link href="/app">Projects</Link></div>
          <h1 className="row gap wrap">{project.name} {project.sampleId && <SyntheticBadge />}</h1>
          {r && <p className="muted small" style={{ margin: 0 }}>{r.files.length} source file{r.files.length === 1 ? '' : 's'} · {r.files.reduce((s, f) => s + f.lines, 0).toLocaleString()} lines · {fmtPct(coveragePct(r.coverage))} API coverage{r.depsName ? ` · ${r.depsName}` : ''}</p>}
        </div>
        <div className="toolbar">
          <label className="field inline"><span>Target</span>
            <select value={project.target} onChange={e => update({ target: e.target.value })} aria-label="Target GPU">
              {TARGETS.map(t => <option key={t.id} value={t.id}>{t.id} · {t.gpus.split(',')[0]}</option>)}
            </select>
          </label>
          <label className="check" title="Rewrite kernel<<<...>>>(args) as hipLaunchKernelGGL(kernel, ...)"><input type="checkbox" checked={project.ggl} onChange={e => update({ ggl: e.target.checked })} /> hipLaunchKernelGGL</label>
          {!empty && <>
            <button type="button" className="btn outline" onClick={() => saveScan(project, r, update)} title="Save this scan to history (S)">Save scan</button>
            <button type="button" className="btn outline" onClick={() => downloadZip(project, r)} title="Download translated files (D)">Download .zip</button>
            <Link href={`/app/p/${id}/report`} className="btn primary" title="Printable readiness report (P)">Report</Link>
          </>}
          {score !== null && <ScoreRing score={score} />}
        </div>
      </div>
      <nav className="tabs" aria-label="Project sections">
        {TABS.map(([t, l], i) => <Link key={t} href={`/app/p/${id}${t === 'overview' ? '' : '/' + t}`} aria-current={t === tab ? 'page' : undefined} title={`${l} (${i + 1})`}>{l}</Link>)}
      </nav>
      <Suspense fallback={<div className="boot"><span className="spinner" /></div>}>
        {tab === 'sources' || empty
          ? <Sources project={project} ds={ds} update={update} />
          : <Tabs tab={tab} project={project} ds={ds} r={r} update={update} />}
      </Suspense>
      <Modal open={help} onClose={() => setHelp(false)} title="Keyboard shortcuts">
        <dl className="shortcuts">
          <div><dt><kbd>1</kbd>&ndash;<kbd>6</kbd></dt><dd>Switch sections</dd></div>
          <div><dt><kbd>S</kbd></dt><dd>Save a scan to history</dd></div>
          <div><dt><kbd>D</kbd></dt><dd>Download translated files</dd></div>
          <div><dt><kbd>P</kbd></dt><dd>Open the printable report</dd></div>
          <div><dt><kbd>J</kbd> <kbd>K</kbd></dt><dd>Next / previous hazard in a file</dd></div>
          <div><dt><kbd>&#8984;</kbd>/<kbd>Ctrl</kbd> <kbd>Z</kbd></dt><dd>Undo the last delete</dd></div>
        </dl>
      </Modal>
    </section>
  )
}

export function ScoreRing({ score, size = 54 }: { score: number; size?: number }) {
  const r = size / 2 - 5, c = 2 * Math.PI * r
  const tone = score >= 80 ? 'good' : score >= 50 ? 'mid' : 'bad'
  return (
    <div className={`ring ${tone}`} title="Readiness: 100 means no open work items" role="img" aria-label={`Readiness ${score} out of 100`} style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} className="ring-bg" />
        <circle cx={size / 2} cy={size / 2} r={r} className="ring-fg" strokeDasharray={`${(c * score) / 100} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} />
      </svg>
      <span>{score}</span>
    </div>
  )
}
