import { useState } from 'react'
import { TARGETS } from './engine/gfx.ts'
import { deleteProject, getDefaults, getProjects, setDefaults, type Defaults } from './store.ts'
import { exportAll, RAW_SYNC_LIMIT, useSync } from './sync.ts'
import { ask, download, setTheme, toast, useDark } from './ui.tsx'

export default function Settings() {
  const [d, setD] = useState<Defaults>(getDefaults)
  const dark = useDark()
  const sync = useSync()
  const save = (p: Partial<Defaults>) => { const next = { ...d, ...p }; setD(next); setDefaults(next); toast('Saved', { tone: 'ok', ttl: 1500 }) }

  return (
    <section className="page narrow settings">
      <div className="eyebrow">Settings</div>
      <h1>Settings</h1>

      <div className="panel">
        <h2>Defaults for new projects</h2>
        <p className="muted small">Existing projects keep their own target and launch style; change those in the project header.</p>
        <div className="inline-fields">
          <label className="field"><span>Target GPU</span>
            <select value={d.target} onChange={e => save({ target: e.target.value })}>{TARGETS.map(t => <option key={t.id} value={t.id}>{t.id} · {t.gpus}</option>)}</select>
          </label>
          <label className="check"><input type="checkbox" checked={d.ggl} onChange={e => save({ ggl: e.target.checked })} /> Rewrite launches as hipLaunchKernelGGL</label>
          <label className="check"><input type="checkbox" checked={d.wrap} onChange={e => save({ wrap: e.target.checked })} /> Wrap long lines in the diff view</label>
        </div>
      </div>

      <div className="panel">
        <h2>Appearance</h2>
        <div className="seg-control" role="radiogroup" aria-label="Theme">
          <button type="button" role="radio" aria-checked={dark} className={dark ? 'on' : ''} onClick={() => setTheme('dark')}>Dark</button>
          <button type="button" role="radio" aria-checked={!dark} className={!dark ? 'on' : ''} onClick={() => setTheme('light')}>Light</button>
        </div>
      </div>

      <div className="panel">
        <h2>Your data</h2>
        <p>Projects live in this browser (IndexedDB). {sync.user ? <>Signed in as <b>{sync.user.email}</b>: settings, worklist status, scan history and source trees under {(RAW_SYNC_LIMIT / 1e6).toFixed(1)} MB per project sync to your account. Larger trees stay on this device.</> : 'Sign in from the account menu (top right) to sync across devices. It is optional.'}</p>
        <div className="row gap wrap">
          <button type="button" className="btn ghost" onClick={async () => download(`hip-port-export-${new Date().toISOString().slice(0, 10)}.json`, await exportAll())}>Export all my data (JSON)</button>
          <button type="button" className="btn danger-ghost" onClick={async () => {
            const list = getProjects()
            if (!list.length) { toast('Nothing to delete'); return }
            if (!(await ask('Delete all local projects?', `Removes ${list.length} project${list.length > 1 ? 's' : ''} from this browser${sync.user ? ' and from your cloud account' : ''}. You can undo for a few seconds.`, 'Delete everything'))) return
            const undos = await Promise.all(list.map(p => deleteProject(p.id)))
            toast(`Deleted ${list.length} projects`, { undo: () => { for (const u of undos) void u() } })
          }}>Delete all projects</button>
        </div>
      </div>

      <div className="panel">
        <h2>Keyboard shortcuts</h2>
        <dl className="shortcuts">
          <div><dt><kbd>1</kbd>&ndash;<kbd>6</kbd></dt><dd>Switch project sections</dd></div>
          <div><dt><kbd>J</kbd> <kbd>K</kbd></dt><dd>Next / previous hazard in a file</dd></div>
          <div><dt><kbd>S</kbd></dt><dd>Save a scan</dd></div>
          <div><dt><kbd>D</kbd></dt><dd>Download translated files</dd></div>
          <div><dt><kbd>P</kbd></dt><dd>Printable readiness report</dd></div>
          <div><dt><kbd>&#8984;</kbd>/<kbd>Ctrl</kbd> <kbd>Z</kbd></dt><dd>Undo the last delete</dd></div>
          <div><dt><kbd>?</kbd></dt><dd>Show shortcuts on a project page</dd></div>
        </dl>
      </div>
    </section>
  )
}
