import { useEffect, useMemo, useState } from 'react'
import { SAMPLES } from './engine/sample.ts'
import { isDepsFile, scan, snapshotOf, type ScanResult } from './engine/scan.ts'
import { getData, getDefaults, getProjects, saveProject, useProjects, uuid, type Dataset, type Project } from './store.ts'

export function sampleData(id: string): Dataset | null {
  const s = SAMPLES.find(x => x.id === id)
  if (!s) return null
  return { files: s.files.filter(f => !isDepsFile(f.path)), deps: s.files.find(f => isDepsFile(f.path)) ?? null }
}

const DAY = 86400000

export async function createSampleProject(sampleId: string): Promise<Project> {
  const existing = getProjects().find(p => p.sampleId === sampleId)
  if (existing) return existing
  const def = SAMPLES.find(s => s.id === sampleId)!
  const ds = sampleData(sampleId)!
  const r = scan(ds.files, ds.deps, { target: def.target, ggl: false })
  // Two earlier scans so History has progress to show: the first import, then a week of fixes.
  const now = Date.now()
  const firstFixes = r.items.filter(i => i.severity === 'critical' && i.category === 'hazard').slice(0, 2).map(i => i.key)
  const status = Object.fromEntries(firstFixes.map(k => [k, 'done' as const]))
  const snapshots = [
    snapshotOf(r, uuid(), 'First import', now - 14 * DAY),
    snapshotOf(r, uuid(), 'After the first fixes', now - 6 * DAY, k => k in status),
  ]
  return saveProject({
    id: uuid(), name: def.name, createdAt: now - 14 * DAY, updatedAt: now, target: def.target, ggl: false, status, snapshots, sampleId,
    imports: [{ id: uuid(), kind: 'sample', fileName: 'synthetic sample sources', files: def.files.length, skipped: 0, at: now - 14 * DAY }],
  })
}

export async function createBlankProject(name: string): Promise<Project> {
  const now = Date.now()
  const d = getDefaults()
  return saveProject({ id: uuid(), name, createdAt: now, updatedAt: now, target: d.target, ggl: d.ggl, status: {}, snapshots: [], imports: [] })
}

export function useProject(id: string) {
  const list = useProjects()
  const project = list?.find(p => p.id === id) ?? null
  const [ds, setDs] = useState<Dataset | null>(null)
  const importsKey = project ? project.imports.map(i => i.id).join(',') : ''
  useEffect(() => {
    if (!project) return
    let live = true
    if (project.sampleId) { setDs(sampleData(project.sampleId) ?? { files: [], deps: null }); return }
    void getData(project.id).then(d => { if (live) setDs(d ?? { files: [], deps: null }) })
    return () => { live = false }
  }, [project?.id, project?.sampleId, importsKey])
  const update = (patch: Partial<Project>) => project && saveProject({ ...project, ...patch })
  return { loading: list === null || (project !== null && ds === null), project, ds, update }
}

export function useScan(project: Project | null, ds: Dataset | null): ScanResult | null {
  return useMemo(() => project && ds && (ds.files.length || ds.deps) ? scan(ds.files, ds.deps, { target: project.target, ggl: project.ggl }) : null, [ds, project?.target, project?.ggl])
}
