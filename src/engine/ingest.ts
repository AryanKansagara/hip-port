// Turns imported files into project sources: what is kept, what is skipped and why.
import type { Dataset } from '../store.ts'
import { decodeSource } from './decode.ts'
import { isDepsFile, isSource } from './scan.ts'

export const MAX_FILE = 1_000_000
const MAX_TOTAL = 25_000_000
export const SKIP_DIRS = /(^|\/)(\.git|node_modules|build|dist|__pycache__|\.venv|venv)\//

export interface Incoming { path: string; data: Uint8Array; size?: number }
export interface Row { path: string; status: 'added' | 'replaced' | 'skipped'; reason?: string; lines?: number }

/** Folders that belong to a repository's layout, so one of them alone at the top of a zip is not a wrapper. */
const LAYOUT_DIRS = /^(src|source|sources|csrc|cuda|gpu|kernels?|include|includes|inc|lib|libs|common|core|utils?|tests?|examples?|samples?|benchmarks?|apps?|tools|third_party|external)$/i

/** Length of the shared top-level folder (repo-main/...) that zips of GitHub repos and picked folders carry, or 0. */
function wrapperLength(ds: Dataset, incoming: Incoming[]): number {
  const top = incoming[0]?.path.split('/')[0]
  if (!top || !incoming.every(f => f.path.startsWith(top + '/'))) return 0
  return LAYOUT_DIRS.test(top) || ds.files.some(f => f.path.startsWith(top + '/')) ? 0 : top.length + 1
}

/** Classifies incoming files and merges them into the dataset; `wrapped` drops a wrapper folder they all share. Pure apart from TextDecoder. */
export function ingest(ds: Dataset, incoming: Incoming[], wrapped: boolean): { ds: Dataset; rows: Row[] } {
  const files = [...ds.files]
  let deps = ds.deps
  const rows: Row[] = []
  const strip = wrapped ? wrapperLength(ds, incoming) : 0
  let total = files.reduce((s, f) => s + f.text.length, 0)
  for (const f of incoming) {
    const path = f.path.slice(strip)
    // A path someone typed is wanted wherever it is.
    if (wrapped && SKIP_DIRS.test(path)) { rows.push({ path, status: 'skipped', reason: 'Build, VCS or environment folder' }); continue }
    const src = isSource(path), dep = isDepsFile(path)
    if (!src && !dep) { rows.push({ path, status: 'skipped', reason: `Not a CUDA/C++ source or dependency file (${path.includes('.') ? '.' + path.split('.').pop() : 'no extension'})` }); continue }
    const size = f.size ?? f.data.length
    if (size > MAX_FILE) { rows.push({ path, status: 'skipped', reason: `Larger than 1 MB (${(size / 1e6).toFixed(1)} MB); likely generated` }); continue }
    if (!f.data.length) { rows.push({ path, status: 'skipped', reason: 'Empty file' }); continue }
    const { text: raw, remark } = decodeSource(f.data)
    if (raw === undefined) { rows.push({ path, status: 'skipped', reason: remark }); continue }
    if (total + f.data.length > MAX_TOTAL) { rows.push({ path, status: 'skipped', reason: 'Project size limit (25 MB) reached' }); continue }
    const text = raw.replace(/\r\n/g, '\n')
    total += text.length
    const lines = text.split('\n').length
    if (dep) {
      rows.push({ path, status: deps ? 'replaced' : 'added', lines, reason: [deps && deps.path !== path ? `Replaces ${deps.path}` : '', remark].filter(Boolean).join('. ') || undefined })
      deps = { path, text }
      continue
    }
    const i = files.findIndex(x => x.path === path)
    if (i >= 0) { files[i] = { path, text }; rows.push({ path, status: 'replaced', lines, reason: remark }) }
    else { files.push({ path, text }); rows.push({ path, status: 'added', lines, reason: remark }) }
  }
  files.sort((a, b) => a.path.localeCompare(b.path))
  return { ds: { files, deps }, rows }
}
