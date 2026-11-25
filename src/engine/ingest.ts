// Turns imported files into project sources: what is kept, what is skipped and why.
import type { Dataset } from '../store.ts'
import { decodeSource } from './decode.ts'
import { cleanPath } from './paths.ts'
import { isDepsFile, isSource, unclosedRemark } from './scan.ts'

export const MAX_FILE = 1_000_000
const MAX_TOTAL = 25_000_000
export const SKIP_DIRS = /(^|\/)(\.git|node_modules|build|dist|__pycache__|\.venv|venv)\//

export interface Incoming { path: string; data: Uint8Array; size?: number }
export interface Row { path: string; status: 'added' | 'replaced' | 'skipped'; reason?: string; lines?: number }

/** Folders that belong to a repository's layout, so one of them alone at the top of a zip is not a wrapper. */
const LAYOUT_DIRS = /^(src|source|sources|csrc|cuda|gpu|kernels?|include|includes|inc|lib|libs|common|core|utils?|tests?|examples?|samples?|benchmarks?|apps?|tools|third_party|external)$/i

/** Length of the shared top-level folder (repo-main/...) that zips of GitHub repos and picked folders carry, or 0. */
function wrapperLength(ds: Dataset, paths: string[]): number {
  const top = paths[0]?.split('/')[0]
  if (!top || !paths.every(p => p.startsWith(top + '/'))) return 0
  return LAYOUT_DIRS.test(top) || ds.files.some(f => f.path.startsWith(top + '/')) ? 0 : top.length + 1
}

/** A sign-in or error page saved under a source name: a doctype or <html> first, after any XML declaration and comments. */
function isHtml(text: string): boolean {
  let at = 0
  for (;;) {
    while (at < text.length && /\s/.test(text[at])) at++
    const close = text.startsWith('<?xml', at) ? '?>' : text.startsWith('<!--', at) ? '-->' : null
    if (!close) return /^<(?:!doctype\s+html|html)\b/i.test(text.slice(at, at + 200))
    const end = text.indexOf(close, at + 4)
    if (end < 0) return false
    at = end + close.length
  }
}

/** Classifies incoming files and merges them into the dataset; `wrapped` drops a wrapper folder they all share. Pure apart from TextDecoder. */
export function ingest(ds: Dataset, incoming: Incoming[], wrapped: boolean): { ds: Dataset; rows: Row[] } {
  const files = [...ds.files]
  let deps = ds.deps
  const rows: Row[] = []
  const entries = incoming.map(f => ({ ...f, clean: cleanPath(f.path) }))
  const strip = wrapped ? wrapperLength(ds, entries.flatMap(f => f.clean ?? [])) : 0
  let total = files.reduce((s, f) => s + f.text.length, 0)
  for (const f of entries) {
    if (f.clean === null) { rows.push({ path: f.path, status: 'skipped', reason: 'Path leads out of the project folder' }); continue }
    const path = f.clean.slice(strip)
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
    if (isHtml(text)) { rows.push({ path, status: 'skipped', reason: 'HTML page, not source code; download the raw file' }); continue }
    total += text.length
    const lines = text.split('\n').length
    if (dep) {
      rows.push({ path, status: deps ? 'replaced' : 'added', lines, reason: [deps && deps.path !== path ? `Replaces ${deps.path}` : '', remark].filter(Boolean).join('. ') || undefined })
      deps = { path, text }
      continue
    }
    const remarks = [remark, unclosedRemark(text)].filter(Boolean).join('. ') || undefined
    const i = files.findIndex(x => x.path === path)
    if (i >= 0) { files[i] = { path, text }; rows.push({ path, status: 'replaced', lines, reason: remarks }) }
    else { files.push({ path, text }); rows.push({ path, status: 'added', lines, reason: remarks }) }
  }
  files.sort((a, b) => a.path.localeCompare(b.path))
  return { ds: { files, deps }, rows }
}
