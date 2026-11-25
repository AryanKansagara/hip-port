import { useSyncExternalStore } from 'react'
import type { ScanSnapshot, SourceFile } from './engine/scan.ts'

/** Raw sources for a project: kept in IndexedDB, separate from the small project record. */
export interface Dataset { files: SourceFile[]; deps: SourceFile | null }

export interface ImportMeta {
  id: string
  kind: 'files' | 'zip' | 'paste' | 'deps' | 'sample' | 'remove'
  fileName: string
  files: number
  skipped: number
  at: number
}

export type ItemStatus = 'done' | 'wontfix'

export interface Project {
  id: string
  name: string
  createdAt: number
  updatedAt: number
  /** gfx target, e.g. gfx942 */
  target: string
  /** Rewrite <<<>>> launches as hipLaunchKernelGGL */
  ggl: boolean
  /** Worklist item status by stable key */
  status: Record<string, ItemStatus>
  /** Saved scans (history) */
  snapshots: ScanSnapshot[]
  imports: ImportMeta[]
  /** Built-in synthetic sample; its sources are regenerated instead of stored. */
  sampleId?: string
  /** True when this device has no copy of the sources (synced from another device). */
  dataMissing?: boolean
}

// ---------- Tiny IndexedDB wrapper (falls back to memory in private modes) ----------

const DB = 'hip-port'
let dbp: Promise<IDBDatabase | null> | null = null
function db(): Promise<IDBDatabase | null> {
  return dbp ??= new Promise(res => {
    try {
      const req = indexedDB.open(DB, 1)
      req.onupgradeneeded = () => {
        req.result.createObjectStore('projects', { keyPath: 'id' })
        req.result.createObjectStore('data')
      }
      req.onsuccess = () => res(req.result)
      req.onerror = () => res(null)
    } catch { res(null) }
  })
}
const mem = { projects: new Map<string, unknown>(), data: new Map<string, unknown>() }

type Store = 'projects' | 'data'

/** Runs one transaction. Resolves once it has committed, so a resolved write is on disk; rejects otherwise. */
function tx<T>(d: IDBDatabase, stores: Store[], mode: IDBTransactionMode, fn: (t: IDBTransaction) => IDBRequest<T> | void): Promise<T | undefined> {
  return new Promise((res, rej) => {
    const failed = (e: unknown) => rej(new Error(`This browser could not ${mode === 'readonly' ? 'read its storage' : 'store the change, so nothing was saved'} (${(e as DOMException | null)?.name ?? 'transaction aborted'}).${mode === 'readonly' ? '' : ' It may be out of space: export or delete projects, then try again.'}`))
    try {
      const t = d.transaction(stores, mode)
      let r: IDBRequest<T> | void
      t.oncomplete = () => res(r ? r.result : undefined)
      t.onabort = () => failed(t.error)
      try { r = fn(t) } catch (e) { failed(e); t.abort() }
    } catch (e) { failed(e) }
  })
}

// The memory copies are only for browsers without IndexedDB. With it open, a failed read or write
// rejects: falling back would lose the write on reload, or show a project without its sources.
async function idbGetAll<T>(store: 'projects'): Promise<T[]> {
  const d = await db()
  return (d ? await tx(d, [store], 'readonly', t => t.objectStore(store).getAll()) : [...mem[store].values()]) as T[]
}
async function idbGet<T>(store: Store, key: string): Promise<T | undefined> {
  const d = await db()
  return (d ? await tx(d, [store], 'readonly', t => t.objectStore(store).get(key)) : mem[store].get(key)) as T | undefined
}
/** Writes a project record, its sources, or both: all of it or none. */
async function idbPut(p: Project | null, data?: { id: string; ds: Dataset }) {
  const d = await db()
  if (!d) {
    if (p) mem.projects.set(p.id, p)
    if (data) mem.data.set(data.id, data.ds)
    return
  }
  await tx(d, ['projects', 'data'], 'readwrite', t => {
    if (p) t.objectStore('projects').put(p)
    if (data) t.objectStore('data').put(data.ds, data.id)
  })
}
async function idbDel(id: string) {
  const d = await db()
  if (!d) { mem.projects.delete(id); mem.data.delete(id); return }
  await tx(d, ['projects', 'data'], 'readwrite', t => { t.objectStore('projects').delete(id); t.objectStore('data').delete(id) })
}

// ---------- In-memory project list with subscribers ----------

let projects: Project[] = []
let loaded = false
const subs = new Set<() => void>()
const emit = () => { projects = [...projects]; subs.forEach(f => f()) }
const changeHooks = new Set<(p: Project) => void>()

export const onProjectChange = (f: (p: Project) => void) => { changeHooks.add(f); return () => changeHooks.delete(f) }

export async function loadProjects() {
  // Storage that opens but cannot be read starts as an empty list; a later reload must not empty one already shown.
  projects = (await idbGetAll<Project>('projects').catch(() => loaded ? projects : [])).sort((a, b) => b.updatedAt - a.updatedAt)
  loaded = true
  emit()
}

export function useProjects() {
  return useSyncExternalStore(f => { subs.add(f); return () => subs.delete(f) }, () => loaded ? projects : null)
}

export const getProjects = () => projects
export const getProject = (id: string) => projects.find(p => p.id === id)

export async function saveProject(p: Project, opts: { touch?: boolean; silent?: boolean; ds?: Dataset } = {}) {
  const next = opts.touch === false ? p : { ...p, updatedAt: Date.now() }
  await idbPut(next, opts.ds && { id: p.id, ds: opts.ds })
  const i = projects.findIndex(x => x.id === p.id)
  if (i >= 0) projects[i] = next
  else projects.unshift(next)
  projects.sort((a, b) => b.updatedAt - a.updatedAt)
  emit()
  if (!opts.silent) changeHooks.forEach(f => f(next))
  return next
}

// Changes run one after another, each on what the one before it stored, so a change made from an
// older view (an undo in a toast, two quick clicks) never carries stale fields back.
let queue: Promise<unknown> = Promise.resolve()
function inTurn<T>(f: () => Promise<T>): Promise<T> {
  const run = queue.then(f)
  queue = run.catch(() => {})
  return run
}

/** Changes a project's current record. Does nothing when the project has been deleted. */
export const changeProject = (id: string, change: (p: Project) => Partial<Project>) => inTurn(async () => {
  const p = getProject(id)
  if (p) await saveProject({ ...p, ...change(p) })
})

/** Changes a project's current sources and its record together; `change` returns null to leave both alone. */
export const changeSources = (id: string, change: (ds: Dataset, p: Project) => { ds: Dataset; patch: Partial<Project> } | null) => inTurn(async () => {
  const ds = await getData(id) ?? { files: [], deps: null }
  const p = getProject(id)
  const c = p && change(ds, p)
  if (!c) return
  await saveProject({ ...p, ...c.patch }, { ds: c.ds })
})

/** Deletes locally; returns an undo function that restores the project and its data. */
export async function deleteProject(id: string): Promise<() => Promise<void>> {
  const p = getProject(id)
  const data = await idbGet<Dataset>('data', id)
  await idbDel(id)
  projects = projects.filter(x => x.id !== id)
  emit()
  addTombstone(id)
  return async () => {
    if (!p) return
    await saveProject(p, { ds: data })
    removeTombstone(id)
  }
}

export const getData = (id: string) => idbGet<Dataset>('data', id)
export const putData = (id: string, ds: Dataset) => idbPut(null, { id, ds })

export const uuid = () => crypto.randomUUID()

// Tombstones keep a project deleted offline from coming back on the next sync.
const TOMB = 'hip-port.tombstones'
export const tombstones = (): string[] => { try { return JSON.parse(localStorage.getItem(TOMB) ?? '[]') } catch { return [] } }
const addTombstone = (id: string) => localStorage.setItem(TOMB, JSON.stringify([...new Set([...tombstones(), id])]))
const removeTombstone = (id: string) => localStorage.setItem(TOMB, JSON.stringify(tombstones().filter(x => x !== id)))
export const clearTombstones = (ids: string[]) => localStorage.setItem(TOMB, JSON.stringify(tombstones().filter(x => !ids.includes(x))))

// ---------- App defaults (settings page) ----------

export interface Defaults { target: string; ggl: boolean; wrap: boolean }
const DEF_KEY = 'hip-port.defaults'
export function getDefaults(): Defaults {
  const base: Defaults = { target: 'gfx942', ggl: false, wrap: false }
  try { return { ...base, ...JSON.parse(localStorage.getItem(DEF_KEY) ?? '{}') } } catch { return base }
}
export const setDefaults = (d: Defaults) => localStorage.setItem(DEF_KEY, JSON.stringify(d))
