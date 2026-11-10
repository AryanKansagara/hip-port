import { useRef, useState } from 'react'
import { isDepsFile, isSource, scan, snapshotOf, type SourceFile } from './engine/scan.ts'
import { readZip } from './engine/zip.ts'
import { resolver } from './Project.tsx'
import { sampleData } from './projectData.ts'
import { putData, saveProject, uuid, type Dataset, type ImportMeta, type Project } from './store.ts'
import { go, Link, toast } from './ui.tsx'

const MAX_FILE = 1_000_000
const MAX_TOTAL = 25_000_000
const SKIP_DIRS = /(^|\/)(\.git|node_modules|build|dist|__pycache__|\.venv|venv)\//

export interface Row { path: string; status: 'added' | 'replaced' | 'skipped'; reason?: string; lines?: number }

/** Classifies incoming files and merges them into the dataset. Pure apart from TextDecoder. */
export function ingest(ds: Dataset, incoming: { path: string; data: Uint8Array }[]): { ds: Dataset; rows: Row[] } {
  const files = [...ds.files]
  let deps = ds.deps
  const rows: Row[] = []
  // Strip a shared top-level folder (repo-main/...) that zips of GitHub repos carry.
  const tops = new Set(incoming.map(f => f.path.split('/')[0]))
  const strip = tops.size === 1 && incoming.every(f => f.path.includes('/')) ? [...tops][0].length + 1 : 0
  let total = files.reduce((s, f) => s + f.text.length, 0)
  const dec = new TextDecoder('utf-8', { fatal: false })
  for (const f of incoming) {
    const path = f.path.slice(strip)
    if (SKIP_DIRS.test(path)) { rows.push({ path, status: 'skipped', reason: 'Build, VCS or environment folder' }); continue }
    const src = isSource(path), dep = isDepsFile(path)
    if (!src && !dep) { rows.push({ path, status: 'skipped', reason: `Not a CUDA/C++ source or dependency file (${path.includes('.') ? '.' + path.split('.').pop() : 'no extension'})` }); continue }
    if (f.data.length > MAX_FILE) { rows.push({ path, status: 'skipped', reason: `Larger than 1 MB (${(f.data.length / 1e6).toFixed(1)} MB); likely generated` }); continue }
    if (!f.data.length) { rows.push({ path, status: 'skipped', reason: 'Empty file' }); continue }
    if (f.data.subarray(0, 4096).includes(0)) { rows.push({ path, status: 'skipped', reason: 'Binary content' }); continue }
    if (total + f.data.length > MAX_TOTAL) { rows.push({ path, status: 'skipped', reason: 'Project size limit (25 MB) reached' }); continue }
    const text = dec.decode(f.data).replace(/\r\n/g, '\n')
    total += text.length
    const lines = text.split('\n').length
    if (dep) {
      rows.push({ path, status: deps ? 'replaced' : 'added', lines, reason: deps && deps.path !== path ? `Replaces ${deps.path}` : undefined })
      deps = { path, text }
      continue
    }
    const i = files.findIndex(x => x.path === path)
    if (i >= 0) { files[i] = { path, text }; rows.push({ path, status: 'replaced', lines }) }
    else { files.push({ path, text }); rows.push({ path, status: 'added', lines }) }
  }
  files.sort((a, b) => a.path.localeCompare(b.path))
  return { ds: { files, deps }, rows }
}

export default function Sources({ project, ds, update }: { project: Project; ds: Dataset; update(p: Partial<Project>): void }) {
  const [rows, setRows] = useState<Row[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pasteName, setPasteName] = useState('kernel.cu')
  const [paste, setPaste] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement>(null)
  const empty = !ds.files.length && !ds.deps

  const commit = async (next: Dataset, r: Row[], meta: Omit<ImportMeta, 'id' | 'at' | 'files' | 'skipped'>) => {
    const added = r.filter(x => x.status !== 'skipped').length
    setRows(r)
    if (!added) { setError('Nothing was imported. See the reasons below.'); return }
    setError(null)
    await putData(project.id, next)
    const res = scan(next.files, next.deps, { target: project.target, ggl: project.ggl })
    const snap = snapshotOf(res, uuid(), `Import: ${meta.fileName}`.slice(0, 60), Date.now(), resolver(project))
    await saveProject({ ...project, dataMissing: false, snapshots: [...project.snapshots, snap], imports: [...project.imports, { ...meta, id: uuid(), at: Date.now(), files: added, skipped: r.length - added }] })
    toast(`Imported ${added} file${added > 1 ? 's' : ''}. Readiness ${snap.score}.`, { tone: 'ok', action: { label: 'See overview', run: () => go(`/app/p/${project.id}`) } })
  }

  const handleFiles = async (list: FileList | File[]) => {
    setBusy(true); setError(null)
    try {
      const incoming: { path: string; data: Uint8Array }[] = []
      const zips: string[] = []
      for (const f of Array.from(list)) {
        const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
        const data = new Uint8Array(await f.arrayBuffer())
        if (/\.zip$/i.test(f.name)) {
          zips.push(f.name)
          if (data.length > 200_000_000) throw new Error(`${f.name} is larger than 200 MB. Zip just the source folders.`)
          incoming.push(...(await readZip(data)).map(e => ({ path: e.path, data: e.data })))
        } else incoming.push({ path: rel, data })
      }
      const { ds: next, rows: r } = ingest(ds, incoming)
      await commit(next, r, { kind: zips.length ? 'zip' : 'files', fileName: zips[0] ?? (list.length === 1 ? Array.from(list)[0].name : `${list.length} files`) })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally { setBusy(false) }
  }

  const addPaste = async () => {
    const name = pasteName.trim() || 'kernel.cu'
    if (!paste.trim()) { setError('Paste some code first.'); return }
    const { ds: next, rows: r } = ingest(ds, [{ path: name, data: new TextEncoder().encode(paste) }])
    await commit(next, r, { kind: isDepsFile(name) ? 'deps' : 'paste', fileName: name })
    setPaste('')
  }

  const remove = async (path: string) => {
    const prev = ds
    const next: Dataset = path === ds.deps?.path ? { ...ds, deps: null } : { ...ds, files: ds.files.filter(f => f.path !== path) }
    // A log entry also tells the project view to reload its sources.
    const log = (fileName: string) => ({ id: uuid(), kind: 'remove' as const, fileName, files: 1, skipped: 0, at: Date.now() })
    await putData(project.id, next)
    update({ imports: [...project.imports, log(path)] })
    toast(`Removed ${path}`, { undo: async () => { await putData(project.id, prev); update({ imports: [...project.imports, log(`${path} (restored)`)] }) } })
  }

  const makeCopy = async () => {
    const d = sampleData(project.sampleId!)!
    const now = Date.now()
    const p = await saveProject({ ...project, id: uuid(), name: `${project.name} (editable copy)`, sampleId: undefined, createdAt: now, snapshots: [...project.snapshots], status: { ...project.status }, imports: [] })
    await putData(p.id, d)
    await saveProject({ ...p, imports: [{ id: uuid(), kind: 'sample', fileName: 'copy of synthetic sample', files: d.files.length + (d.deps ? 1 : 0), skipped: 0, at: now }] })
    go(`/app/p/${p.id}/sources`)
  }

  if (project.sampleId) return (
    <div className="stack">
      <div className="callout"><p><b>This is a synthetic sample.</b> Its sources were written for HIP Port and are regenerated each time, so they cannot be edited here. Make a copy to add or remove files, or start a new project with your own code.</p>
        <div className="row gap wrap"><button type="button" className="btn primary" onClick={makeCopy}>Make an editable copy</button><Link href="/app" className="btn ghost">New project</Link></div></div>
      <SourceList ds={ds} />
    </div>
  )

  return (
    <div className="stack">
      {empty && <div className="callout"><p><b>Add your sources to start.</b> Everything is read in this browser. See <Link href="/docs">the docs</Link> for how to bundle a repo and which files matter.</p></div>}
      <div
        className={`dropzone${drag ? ' over' : ''}${busy ? ' busy' : ''}`}
        onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files.length) void handleFiles(e.dataTransfer.files) }}
      >
        <div className="dz-icon" aria-hidden="true">{'{ }'}</div>
        <h3>{busy ? 'Reading files…' : 'Drop CUDA sources, a repo .zip, or requirements.txt / environment.yml'}</h3>
        <p className="muted small">.cu .cuh .cpp .cc .h .hpp .inl · zip archives (deflate or stored) · 1 MB per file, 25 MB per project</p>
        <div className="row gap wrap center-x">
          <button type="button" className="btn primary" disabled={busy} onClick={() => fileRef.current?.click()}>Choose files or zip</button>
          <button type="button" className="btn outline" disabled={busy} onClick={() => dirRef.current?.click()}>Choose a folder</button>
        </div>
        <input ref={fileRef} type="file" multiple hidden accept=".cu,.cuh,.cpp,.cc,.cxx,.c,.h,.hpp,.hh,.hxx,.inl,.zip,.txt,.yml,.yaml" onChange={e => { if (e.target.files?.length) void handleFiles(e.target.files); e.target.value = '' }} />
        <input ref={dirRef} type="file" multiple hidden {...{ webkitdirectory: '' }} onChange={e => { if (e.target.files?.length) void handleFiles(e.target.files); e.target.value = '' }} />
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {rows && <ValidationReport rows={rows} />}

      <div className="panel">
        <h2>Paste code</h2>
        <div className="paste">
          <label className="field"><span>File name</span><input value={pasteName} onChange={e => setPasteName(e.target.value)} placeholder="kernel.cu or requirements.txt" /></label>
          <label className="field"><span>Contents</span><textarea value={paste} onChange={e => setPaste(e.target.value)} rows={8} spellCheck={false} placeholder={'__global__ void k(float* x) {\n  int lane = threadIdx.x % 32;\n}'} className="mono" /></label>
          <div><button type="button" className="btn outline" onClick={() => void addPaste()}>Add to project</button></div>
        </div>
      </div>
      <SourceList ds={ds} onRemove={p => void remove(p)} />
      {project.imports.length > 0 && (
        <div className="panel">
          <h2>Import log</h2>
          <ul className="log">{[...project.imports].reverse().map(i => <li key={i.id}><span className="muted">{new Date(i.at).toLocaleString()}</span> {i.kind === 'remove' ? <>removed <b>{i.fileName}</b></> : <>{i.kind} · <b>{i.fileName}</b> · {i.files} imported{i.skipped ? `, ${i.skipped} skipped` : ''}</>}</li>)}</ul>
        </div>
      )}
    </div>
  )
}

function ValidationReport({ rows }: { rows: Row[] }) {
  const skipped = rows.filter(r => r.status === 'skipped')
  const ok = rows.length - skipped.length
  const [all, setAll] = useState(false)
  const shown = all ? rows : [...skipped.slice(0, 30), ...rows.filter(r => r.status !== 'skipped').slice(0, 10)]
  return (
    <div className="panel" aria-live="polite">
      <h2>Validation report</h2>
      <p><b className="good-text">{ok} imported</b> · <b className={skipped.length ? 'warn-text' : 'muted'}>{skipped.length} skipped</b> of {rows.length} files</p>
      <div className="table-wrap"><table className="data">
        <thead><tr><th>File</th><th>Result</th><th>Detail</th></tr></thead>
        <tbody>{shown.map((r, i) => <tr key={i}><td className="mono small">{r.path}</td><td><span className={`pill ${r.status === 'skipped' ? 'mid' : 'good'}`}>{r.status}</span></td><td className="small">{r.reason ?? (r.lines ? `${r.lines} lines` : '')}</td></tr>)}</tbody>
      </table></div>
      {shown.length < rows.length && <button type="button" className="linkish small" onClick={() => setAll(true)}>Show all {rows.length}</button>}
    </div>
  )
}

function SourceList({ ds, onRemove }: { ds: Dataset; onRemove?(path: string): void }) {
  const all: (SourceFile & { dep?: boolean })[] = [...ds.files, ...(ds.deps ? [{ ...ds.deps, dep: true }] : [])]
  if (!all.length) return null
  return (
    <div className="panel">
      <h2>Sources <small className="muted">{all.length} file{all.length > 1 ? 's' : ''} · {(all.reduce((s, f) => s + f.text.length, 0) / 1024).toFixed(0)} KB</small></h2>
      <ul className="src-list">
        {all.map(f => (
          <li key={f.path}>
            <span className="mono">{f.path}</span>
            <span className="muted small">{f.dep ? 'Python dependencies' : `${f.text.split('\n').length} lines`}</span>
            {onRemove && <button type="button" className="btn danger-ghost sm" onClick={() => onRemove(f.path)} aria-label={`Remove ${f.path}`}>Remove</button>}
          </li>
        ))}
      </ul>
    </div>
  )
}
