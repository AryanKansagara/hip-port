import { useRef, useState } from 'react'
import { ingest, MAX_FILE, SKIP_DIRS, type Incoming, type Row } from './engine/ingest.ts'
import { isDepsFile, isSource, scan, snapshotOf, type SourceFile } from './engine/scan.ts'
import { readZip } from './engine/zip.ts'
import { resolver } from './Project.tsx'
import { sampleData } from './projectData.ts'
import { changeSources, putData, saveProject, uuid, type Dataset, type ImportMeta, type Project } from './store.ts'
import { go, Link, toast } from './ui.tsx'

export default function Sources({ project, ds }: { project: Project; ds: Dataset }) {
  const [rows, setRows] = useState<Row[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pasteName, setPasteName] = useState('kernel.cu')
  const [paste, setPaste] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement>(null)
  const empty = !ds.files.length && !ds.deps

  const commit = async (incoming: Incoming[], wrapped: boolean, meta: Omit<ImportMeta, 'id' | 'at' | 'files' | 'skipped'>) => {
    let rows: Row[] = [], added = 0, score = 0
    await changeSources(project.id, (cur, p) => {
      const { ds: next, rows: r } = ingest(cur, incoming, wrapped)
      rows = r
      added = r.filter(x => x.status !== 'skipped').length
      if (!added) return null
      const res = scan(next.files, next.deps, { target: p.target, ggl: p.ggl })
      const snap = snapshotOf(res, uuid(), `Import: ${meta.fileName}`.slice(0, 60), Date.now(), resolver(p))
      score = snap.score
      return { ds: next, patch: { dataMissing: false, snapshots: [...p.snapshots, snap], imports: [...p.imports, { ...meta, id: uuid(), at: Date.now(), files: added, skipped: r.length - added }] } }
    })
    setRows(rows)
    if (!added) { setError('Nothing was imported. See the reasons below.'); return }
    setError(null)
    toast(`Imported ${added} file${added > 1 ? 's' : ''}. Readiness ${score}.`, { tone: 'ok', action: { label: 'See overview', run: () => go(`/app/p/${project.id}`) } })
  }

  const handleFiles = async (list: FileList | File[]) => {
    // Copy first: the input's FileList is live and empties when its value is reset.
    const all = Array.from(list)
    setBusy(true); setError(null); setRows(null)
    try {
      // Entries an import skips anyway are listed without being decompressed; `size` is what the zip declares.
      const incoming: Incoming[] = []
      const zips: string[] = []
      for (const f of all) {
        const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
        const data = new Uint8Array(await f.arrayBuffer())
        if (/\.zip$/i.test(f.name)) {
          zips.push(f.name)
          if (data.length > 200_000_000) throw new Error(`${f.name} is larger than 200 MB. Zip just the source folders.`)
          const wanted = (p: string, size: number) => (isSource(p) || isDepsFile(p)) && size <= MAX_FILE && !SKIP_DIRS.test(p.slice(p.indexOf('/') + 1))
          const skip: typeof incoming = []
          const read = await readZip(data, wanted, (path, size) => skip.push({ path, data: new Uint8Array(0), size }))
          incoming.push(...read, ...skip)
        } else incoming.push({ path: rel, data })
      }
      await commit(incoming, true, { kind: zips.length ? 'zip' : 'files', fileName: zips[0] ?? (all.length === 1 ? all[0].name : `${all.length} files`) })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally { setBusy(false) }
  }

  const addPaste = async () => {
    const name = pasteName.trim() || 'kernel.cu'
    if (!paste.trim()) { setError('Paste some code first.'); return }
    await commit([{ path: name, data: new TextEncoder().encode(paste) }], false, { kind: isDepsFile(name) ? 'deps' : 'paste', fileName: name })
    setPaste('')
  }

  const remove = async (path: string) => {
    const dep = path === ds.deps?.path
    const file = dep ? ds.deps : ds.files.find(f => f.path === path)
    if (!file) return
    // A log entry also tells the project view to reload its sources.
    const log = (fileName: string) => ({ id: uuid(), kind: 'remove' as const, fileName, files: 1, skipped: 0, at: Date.now() })
    await changeSources(project.id, (cur, p) => ({
      ds: dep ? { ...cur, deps: cur.deps?.path === path ? null : cur.deps } : { ...cur, files: cur.files.filter(f => f.path !== path) },
      patch: { imports: [...p.imports, log(path)] },
    }))
    // Only this file comes back, and only if nothing has been imported in its place since.
    const restore = () => changeSources(project.id, (cur, p) => {
      if (dep ? cur.deps : cur.files.some(f => f.path === path)) return null
      return {
        ds: dep ? { ...cur, deps: file } : { ...cur, files: [...cur.files, file].sort((a, b) => a.path.localeCompare(b.path)) },
        patch: { imports: [...p.imports, log(`${path} (restored)`)] },
      }
    })
    toast(`Removed ${path}`, { undo: () => void restore() })
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
