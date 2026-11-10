import { useEffect, useState } from 'react'
import { SAMPLES } from './engine/sample.ts'
import { createBlankProject, createSampleProject } from './projectData.ts'
import { deleteProject, saveProject, useProjects, type Project } from './store.ts'
import { useSync } from './sync.ts'
import { ask, go, Link, SyntheticBadge, toast } from './ui.tsx'

export default function Workspace() {
  const list = useProjects()
  const sync = useSync()
  const [busy, setBusy] = useState<string | null>(null)

  // /app?sample=mini-repo opens (or creates) the sample straight away.
  useEffect(() => {
    const q = new URLSearchParams(location.search)
    const s = q.get('sample')
    const tab = q.get('tab')?.replace(/[^a-z]/g, '')
    if (!s || list === null || busy) return
    if (!SAMPLES.some(x => x.id === s)) return
    setBusy(s)
    void createSampleProject(s).then(p => go(`/app/p/${p.id}${tab ? '/' + tab : ''}`, { replace: true }))
  }, [list, busy])

  const openSample = async (id: string) => { setBusy(id); const p = await createSampleProject(id); go(`/app/p/${p.id}`) }
  const newProject = async () => {
    const p = await createBlankProject(`Port ${(list?.filter(p => !p.sampleId).length ?? 0) + 1}`)
    go(`/app/p/${p.id}/sources`)
  }

  if (list === null || busy) return <div className="boot" role="status"><span className="spinner" /> {busy ? 'Scanning the synthetic sample' : 'Opening your projects'}</div>

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">Workspace</div>
          <h1>Projects</h1>
          <p className="muted">One project per codebase you are porting. Each keeps its sources, gfx target, worklist status and scan history. {sync.user ? 'Synced to your account.' : 'Stored on this device.'}</p>
        </div>
        {list.length > 0 && <button type="button" className="btn primary" onClick={newProject}>New project</button>}
      </div>

      {list.length === 0 ? (
        <div className="onboard">
          <article className="onboard-card feature-card">
            <div className="eyebrow">Two minutes</div>
            <h2>Try the sample</h2>
            <p>Synthetic CUDA written for this tool, with the bugs real ports hit: warp-size assumptions, inline PTX, texture references, a CUDA-pinned Python stack.</p>
            <div className="stack-sm">
              {SAMPLES.map(s => (
                <button key={s.id} type="button" className="sample-btn" onClick={() => void openSample(s.id)}>
                  <span><strong>{s.name}</strong><small>{s.blurb}</small></span><span aria-hidden="true">&rarr;</span>
                </button>
              ))}
            </div>
          </article>
          <article className="onboard-card">
            <div className="eyebrow">Your code</div>
            <h2>Import your code</h2>
            <p>Drop .cu, .cuh, .cpp and header files, a zip of the repo, or paste a kernel. Add requirements.txt or environment.yml to check the Python side. Every skipped file is listed with the reason.</p>
            <div className="row gap wrap">
              <button type="button" className="btn primary" onClick={newProject}>Import your code</button>
              <Link href="/docs" className="btn ghost">How to export</Link>
            </div>
            <p className="muted small">Files are read in your browser and stay on this device unless you sign in to sync.</p>
          </article>
        </div>
      ) : (
        <>
          <div className="project-grid">{list.map(p => <ProjectCard key={p.id} p={p} />)}</div>
          <div className="samples-row">
            <span className="muted small">Samples:</span>
            {SAMPLES.map(s => <button key={s.id} type="button" className="linkish small" onClick={() => void openSample(s.id)}>{s.name}</button>)}
          </div>
        </>
      )}
    </section>
  )
}

function ProjectCard({ p }: { p: Project }) {
  const [renaming, setRenaming] = useState(false)
  const last = p.snapshots.at(-1)
  const prev = p.snapshots.at(-2)
  return (
    <article className="project-card">
      <div className="row between">
        {renaming ? (
          <form onSubmit={e => { e.preventDefault(); const v = String(new FormData(e.currentTarget).get('n')).trim(); if (v) void saveProject({ ...p, name: v }); setRenaming(false) }}>
            <input name="n" defaultValue={p.name} autoFocus aria-label="Project name" onBlur={e => e.currentTarget.form?.requestSubmit()} />
          </form>
        ) : <h3><Link href={`/app/p/${p.id}`}>{p.name}</Link></h3>}
        {p.sampleId && <SyntheticBadge />}
      </div>
      <dl className="mini-stats">
        <div><dt>Target</dt><dd className="mono">{p.target}</dd></div>
        <div><dt>Readiness</dt><dd>{last ? <>{last.score}{prev ? <small className={last.score >= prev.score ? 'good-text' : 'error'}> {last.score >= prev.score ? '+' : ''}{last.score - prev.score}</small> : null}</> : '—'}</dd></div>
        <div><dt>Critical</dt><dd>{last ? last.counts.critical : '—'}</dd></div>
        <div><dt>Scans</dt><dd>{p.snapshots.length}</dd></div>
      </dl>
      {p.dataMissing && <p className="warn-text small">The sources live on another device. Re-import here to rescan.</p>}
      <div className="row between">
        <span className="muted small">Updated {new Date(p.updatedAt).toLocaleDateString()}</span>
        <div className="row gap-sm">
          <button type="button" className="btn ghost sm" onClick={() => setRenaming(true)}>Rename</button>
          <button type="button" className="btn danger-ghost sm" onClick={async () => {
            if (!(await ask(`Delete “${p.name}”?`, 'Removes the project, its sources and its scan history from this device (and the cloud, if you are signed in). You can undo for a few seconds.', 'Delete project'))) return
            const undo = await deleteProject(p.id)
            toast(`Deleted ${p.name}`, { undo: () => void undo() })
          }}>Delete</button>
        </div>
      </div>
    </article>
  )
}
