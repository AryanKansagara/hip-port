import { useEffect } from 'react'
import { SEVERITIES } from './engine/hazards.ts'
import { coveragePct, readiness, STATUS_LABEL, worklistCsv } from './engine/scan.ts'
import { resolver, slug } from './Project.tsx'
import { useProject, useScan } from './projectData.ts'
import { SEV_LABEL } from './Tabs.tsx'
import { download, fmtPct, Link, SyntheticBadge } from './ui.tsx'

export default function Report({ id }: { id: string }) {
  const { loading, project, ds } = useProject(id)
  const r = useScan(project, ds)
  useEffect(() => { if (project) document.title = `ROCm readiness report · ${project.name}` }, [project?.name])
  if (loading) return <div className="boot"><span className="spinner" /> Building the report</div>
  if (!project || !r) return <section className="page narrow center"><h1>Nothing to report</h1><p className="muted">The project is missing or has no sources yet.</p><Link href="/app" className="btn primary">All projects</Link></section>

  const done = resolver(project)
  const open = r.items.filter(i => !done(i.key))
  const score = readiness(r.items, done)
  const count = (s: string) => open.filter(i => i.severity === s).length
  const first = project.snapshots[0]
  const t = r.target

  return (
    <article className="report">
      <div className="report-actions">
        <Link href={`/app/p/${id}`} className="btn ghost">Back</Link>
        <button type="button" className="btn outline" onClick={() => download(`${slug(project.name)}-worklist.csv`, worklistCsv(r.items, k => project.status[k] ?? 'open'))}>Worklist CSV</button>
        <button type="button" className="btn primary" onClick={() => print()}>Print or save PDF</button>
      </div>
      <header className="report-head">
        <div className="eyebrow">ROCm readiness report</div>
        <h1>{project.name} {project.sampleId && <SyntheticBadge />}</h1>
        <p className="muted">Target {t.id} ({t.gpus}) · wave{t.wave} · generated {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</p>
      </header>

      <div className="kpis">
        <div className="kpi"><span>Readiness</span><b>{score}<small>/100</small></b>{first && <em className="muted">{first.score} at “{first.label}”</em>}</div>
        <div className="kpi"><span>API coverage</span><b>{fmtPct(coveragePct(r.coverage))}</b><em className="muted">{r.coverage.total} CUDA references</em></div>
        <div className="kpi"><span>Open critical / high</span><b>{count('critical')} / {count('high')}</b><em className="muted">{open.length} open in total</em></div>
        <div className="kpi"><span>Sources</span><b>{r.files.length}</b><em className="muted">{r.files.reduce((s, f) => s + f.lines, 0).toLocaleString()} lines</em></div>
      </div>

      <h2>Summary</h2>
      <ul>
        <li>{r.coverage.converted} of {r.coverage.total} CUDA API references translate mechanically; {r.coverage.partial} translate but need review; {r.coverage.unsupported + r.coverage.unmapped} need a hand port.</li>
        <li>{count('critical')} critical item{count('critical') === 1 ? '' : 's'} will either fail to compile under hipcc or compute wrong results on {t.id}.</li>
        {r.libs.length > 0 && <li>Libraries: {r.libs.map(l => `${l.cuda} → ${l.rocm}`).join('; ')}.</li>}
        {r.deps && <li>Python: {r.deps.deps.length} packages checked; {r.deps.deps.filter(d => d.status === 'cuda-only').length} CUDA-only, {r.deps.deps.filter(d => d.status === 'fork' || d.status === 'partial').length} need a fork, source build or caveat.</li>}
      </ul>

      <h2>Readiness checklist</h2>
      <ul className="checklist">{r.checklist.map(c => <li key={c.id} className={c.state}><i /><div><b>{c.title}</b><p className="small">{c.detail}</p>{c.command && <code className="small">{c.command}</code>}</div></li>)}</ul>

      <h2>Open work items</h2>
      {SEVERITIES.map(s => {
        const list = open.filter(i => i.severity === s)
        if (!list.length) return null
        return (
          <section key={s} className="report-sev">
            <h3><span className={`sev ${s}`}>{SEV_LABEL[s]}</span> {list.length}</h3>
            <table className="data"><tbody>{list.map(i => <tr key={i.key}><td className="mono small">{i.file}{i.line ? `:${i.line}` : ''}</td><td><b>{i.title}</b><div className="small">{i.fix}</div></td></tr>)}</tbody></table>
          </section>
        )
      })}

      {r.deps && <>
        <h2>Python dependencies</h2>
        <table className="data"><thead><tr><th>Package</th><th>Status</th><th>Action</th></tr></thead>
          <tbody>{r.deps.deps.filter(d => d.status !== 'ok').map((d, i) => <tr key={i}><td className="mono">{d.name}{d.spec ? ` ${d.spec}` : ''}</td><td>{STATUS_LABEL[d.status]}</td><td className="small">{d.action}</td></tr>)}</tbody></table>
        <p className="muted small">Package table checked October 2026.</p>
      </>}

      {project.snapshots.length > 1 && <>
        <h2>Progress</h2>
        <table className="data"><thead><tr><th>Scan</th><th>Date</th><th className="num">Critical</th><th className="num">High</th><th className="num">Readiness</th></tr></thead>
          <tbody>{project.snapshots.map(s => <tr key={s.id}><td>{s.label}</td><td>{new Date(s.at).toLocaleDateString()}</td><td className="num">{s.counts.critical}</td><td className="num">{s.counts.high}</td><td className="num">{s.score}</td></tr>)}</tbody></table>
      </>}
      <p className="muted small report-foot">Produced with HIP Port, an independent tool not affiliated with AMD. Mappings follow the hipify tables for ROCm 6.x/7.0; verify against your ROCm release.</p>
    </article>
  )
}
