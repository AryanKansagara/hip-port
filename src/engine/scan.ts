// Project-level scan: translate every source, collect hazards and library usage, check Python
// dependencies, and turn it all into a severity-ranked worklist and a readiness checklist.
import { detect, SEVERITIES, type Hazard, type Severity, RULE_BY_ID } from './hazards.ts'
import { mask } from './lex.ts'
import { LIBS, type Lib } from './mapping.ts'
import { target as getTarget, type Target } from './gfx.ts'
import { definedNames, hipPath, translate, type Use } from './translate.ts'
import { parseDeps, type DepsResult } from './pydeps.ts'

export interface SourceFile { path: string; text: string }
export interface ScanOptions { target: string; ggl: boolean }

export const isSource = (p: string) => /\.(cu|cuh|cpp|cc|cxx|c|h|hpp|hh|hxx|inl)$/i.test(p)
export const isDepsFile = (p: string) => /(^|\/)(requirements[\w.-]*\.txt|environment[\w.-]*\.ya?ml|conda[\w.-]*\.ya?ml)$/i.test(p)

export interface Coverage { total: number; converted: number; partial: number; unsupported: number; unmapped: number }

export interface FileResult {
  path: string
  outPath: string
  source: string
  translated: string
  hazards: Hazard[]
  uses: Use[]
  launches: number
  addedInclude: boolean
  lines: number
  coverage: Coverage
  /** True when the file has no CUDA at all (plain C++), so it is copied unchanged */
  plain: boolean
}

export interface WorkItem {
  key: string
  severity: Severity
  category: 'hazard' | 'api' | 'header' | 'dependency'
  file: string
  line: number
  title: string
  detail: string
  fix: string
}

export interface LibUse { lib: Lib; cuda: string; rocm: string; refs: number; files: number; worst: Use['status']; note?: string }

export interface ScanResult {
  target: Target
  files: FileResult[]
  libs: LibUse[]
  coverage: Coverage
  counts: Record<Severity, number>
  items: WorkItem[]
  deps: DepsResult | null
  depsName: string | null
  checklist: CheckItem[]
}

const STATUS_RANK: Use['status'][] = ['ok', 'partial', 'deprecated', 'unmapped', 'unsupported']

function coverageOf(uses: Use[]): Coverage {
  const c: Coverage = { total: 0, converted: 0, partial: 0, unsupported: 0, unmapped: 0 }
  for (const u of uses) {
    if (u.kind === 'launch') continue
    const n = u.lines.length
    c.total += n
    if (u.status === 'ok') c.converted += n
    else if (u.status === 'partial' || u.status === 'deprecated') { if (u.hip) c.partial += n; else c.unsupported += n }
    else if (u.status === 'unsupported') c.unsupported += n
    else c.unmapped += n
  }
  return c
}
const addCov = (a: Coverage, b: Coverage): Coverage => ({ total: a.total + b.total, converted: a.converted + b.converted, partial: a.partial + b.partial, unsupported: a.unsupported + b.unsupported, unmapped: a.unmapped + b.unmapped })
/** Share of CUDA references translated mechanically; partial mappings count half. */
export const coveragePct = (c: Coverage) => c.total ? (c.converted + c.partial * 0.5) / c.total : 1

const norm = (s: string) => s.replace(/\s+/g, ' ').trim()

export function scan(files: SourceFile[], depsFile: SourceFile | null, opts: ScanOptions): ScanResult {
  const t = getTarget(opts.target)
  // Names the project defines anywhere are never renamed (a CUDA_CHECK macro, a cudaCheck helper).
  const defined = new Set<string>()
  for (const f of files) for (const n of definedNames(f.text)) defined.add(n)

  const results: FileResult[] = files.filter(f => isSource(f.path)).map(f => {
    const m = mask(f.text)
    const tr = translate(f.text, { ggl: opts.ggl, defined })
    const hazards = detect(f.text, m, t)
    const plain = !tr.uses.length && !hazards.length && !/__global__|__device__|__shared__/.test(m.code)
    return {
      path: f.path, outPath: plain ? f.path : hipPath(f.path), source: f.text, translated: plain ? f.text : tr.out, hazards, uses: tr.uses,
      launches: tr.launches, addedInclude: plain ? false : tr.addedInclude, lines: f.text.split('\n').length, coverage: coverageOf(tr.uses), plain,
    }
  })

  const libMap = new Map<Lib, LibUse & { fileSet: Set<string> }>()
  for (const r of results) for (const u of r.uses) {
    if (u.kind === 'launch') continue
    const e = libMap.get(u.lib) ?? { lib: u.lib, cuda: LIBS[u.lib].cuda, rocm: LIBS[u.lib].rocm, refs: 0, files: 0, worst: 'ok' as Use['status'], note: LIBS[u.lib].note, fileSet: new Set<string>() }
    e.refs += u.lines.length
    e.fileSet.add(r.path)
    if (STATUS_RANK.indexOf(u.status) > STATUS_RANK.indexOf(e.worst)) e.worst = u.status
    libMap.set(u.lib, e)
  }
  const libs = [...libMap.values()].map(({ fileSet, ...l }) => ({ ...l, files: fileSet.size })).sort((a, b) => b.refs - a.refs)

  const deps = depsFile ? parseDeps(depsFile.path, depsFile.text) : null
  const items = workItems(results, deps, depsFile?.path ?? '')
  const counts = Object.fromEntries(SEVERITIES.map(s => [s, items.filter(i => i.severity === s).length])) as Record<Severity, number>
  const coverage = results.reduce((a, r) => addCov(a, r.coverage), coverageOf([]))
  return { target: t, files: results, libs, coverage, counts, items, deps, depsName: depsFile?.path ?? null, checklist: checklist(results, deps, t) }
}

const API_SEV: Record<string, Severity> = { unsupported: 'high', unmapped: 'medium', deprecated: 'medium', partial: 'low' }
const DEP_SEV: Record<string, Severity> = { 'cuda-only': 'high', fork: 'medium', partial: 'medium', rocm: 'low', unknown: 'low' }

export function workItems(files: FileResult[], deps: DepsResult | null, depsPath: string): WorkItem[] {
  const out: WorkItem[] = []
  for (const f of files) {
    for (const h of f.hazards) {
      out.push({ key: `hazard|${f.path}|${h.rule}|${norm(h.snippet)}`, severity: h.severity, category: 'hazard', file: f.path, line: h.line, title: RULE_BY_ID[h.rule].title, detail: h.message, fix: h.fix })
    }
    for (const u of f.uses) {
      const sev = API_SEV[u.status]
      if (!sev) continue
      const what = u.kind === 'header' ? `#include <${u.cuda}>` : u.cuda
      out.push({
        key: `${u.kind === 'header' ? 'header' : 'api'}|${f.path}|${u.cuda}`, severity: sev, category: u.kind === 'header' ? 'header' : 'api', file: f.path, line: u.lines[0],
        title: `${what} ${u.status === 'unmapped' ? 'is not mapped' : u.status === 'unsupported' ? 'has no HIP equivalent' : u.status === 'deprecated' ? 'is deprecated in HIP' : `→ ${u.hip} needs review`}${u.lines.length > 1 ? ` (${u.lines.length} places)` : ''}`,
        detail: u.note ?? '', fix: u.hip && u.status !== 'unsupported' ? `Translated to ${u.hip}. ${u.note ?? ''}`.trim() : 'Port by hand.',
      })
    }
  }
  if (deps) {
    for (const d of deps.deps) {
      let sev = DEP_SEV[d.status]
      if (d.cudaLocal && sev !== 'high') sev = 'high'
      if (!sev) continue
      if (d.status === 'unknown') continue
      out.push({ key: `dependency|${depsPath}|${d.name.toLowerCase()}`, severity: sev, category: 'dependency', file: depsPath, line: d.line, title: `${d.name}${d.cudaLocal ? ` pinned to ${d.cudaLocal}` : ''}: ${STATUS_LABEL[d.status]}`, detail: d.note, fix: d.action })
    }
    for (const x of deps.cudaIndexes) out.push({ key: `dependency|${depsPath}|index|${x.url}`, severity: 'high', category: 'dependency', file: depsPath, line: x.line, title: 'Package index points at CUDA wheels', detail: x.url, fix: 'Replace with the PyTorch ROCm index.' })
    for (const c of deps.cudaChannels) out.push({ key: `dependency|${depsPath}|channel|${c}`, severity: 'medium', category: 'dependency', file: depsPath, line: 0, title: `Conda channel “${c}” only serves CUDA packages`, detail: '', fix: 'Remove the channel.' })
  }
  return out.sort((a, b) => SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity) || a.file.localeCompare(b.file) || a.line - b.line)
}

export const STATUS_LABEL: Record<string, string> = { ok: 'works on ROCm', rocm: 'install the ROCm build', fork: 'needs the ROCm fork or a source build', partial: 'partly supported', 'cuda-only': 'CUDA only', unknown: 'not in table' }

const WEIGHT: Record<Severity, number> = { critical: 12, high: 5, medium: 1.5, low: 0.4 }
/** 0-100. Open items cost points by severity; resolved items cost nothing. */
export function readiness(items: WorkItem[], resolved: (key: string) => boolean = () => false): number {
  const cost = items.filter(i => !resolved(i.key)).reduce((s, i) => s + WEIGHT[i.severity], 0)
  return Math.max(0, Math.round(100 * Math.exp(-cost / 100)))
}

// ---------- Readiness checklist for a PyTorch stack ----------

export interface CheckItem { id: string; state: 'pass' | 'todo' | 'info'; title: string; detail: string; command?: string }

function checklist(files: FileResult[], deps: DepsResult | null, t: Target): CheckItem[] {
  const out: CheckItem[] = []
  const has = (rule: string) => files.some(f => f.hazards.some(h => h.rule === rule))
  const uses = (name: string) => files.some(f => f.uses.some(u => u.cuda === name))
  const dep = (n: string) => deps?.deps.find(d => d.name.toLowerCase().replace(/_/g, '-') === n)

  const torch = dep('torch') ?? dep('pytorch')
  if (!deps) out.push({ id: 'torch', state: 'info', title: 'Add requirements.txt or environment.yml', detail: 'Upload your Python dependency file to check torch, attention libraries and quantization packages.' })
  else if (!torch) out.push({ id: 'torch', state: 'info', title: 'torch not listed', detail: 'No torch dependency found. If it comes from a base image, use rocm/pytorch.' })
  else {
    const bad = torch.cudaLocal || deps.cudaIndexes.length || deps.cudaChannels.length || torch.source === 'conda'
    out.push({ id: 'torch', state: bad ? 'todo' : 'pass', title: 'Install torch from the ROCm wheel index', detail: bad ? `Line ${torch.line} pins a CUDA build${torch.cudaLocal ? ` (+${torch.cudaLocal})` : ''}${deps.cudaIndexes.length ? ' and the file points at a CUDA index' : ''}. Use the ROCm index and the matching torchvision/torchaudio.` : 'torch has no CUDA-specific pin.', command: 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0' })
  }
  if (deps) {
    const cudaOnly = deps.deps.filter(d => d.status === 'cuda-only')
    out.push({ id: 'cuda-only', state: cudaOnly.length ? 'todo' : 'pass', title: 'Remove CUDA-only packages', detail: cudaOnly.length ? cudaOnly.map(d => d.name).join(', ') : 'None found.' })
    const forks = deps.deps.filter(d => d.status === 'fork' || d.status === 'partial')
    out.push({ id: 'forks', state: forks.length ? 'todo' : 'pass', title: 'Switch to ROCm forks and source builds', detail: forks.length ? forks.map(d => d.name).join(', ') : 'None needed.' })
  }
  out.push({ id: 'arch', state: 'info', title: `Build extensions for ${t.id}`, detail: `Set the target before building any C++/HIP extension so you do not compile for every architecture.`, command: `export PYTORCH_ROCM_ARCH=${t.id}` })
  const waveRules = ['warp-size-32', 'shfl-reduction-16', 'mask-32bit', 'ballot-32bit', 'lane-bit-32', 'warp-synchronous']
  const waveHits = files.reduce((s, f) => s + f.hazards.filter(h => waveRules.includes(h.rule)).length, 0)
  out.push({ id: 'wave', state: waveHits ? (t.wave === 64 ? 'todo' : 'info') : 'pass', title: `Kernels are wavefront-size clean (${t.id} runs wave${t.wave})`, detail: waveHits ? `${waveHits} warp-size assumption${waveHits > 1 ? 's' : ''} found.${t.wave === 32 ? ' They happen to work on this wave32 target but break on Instinct GPUs.' : ''}` : 'No 32-lane assumptions found.' })
  const ptx = has('inline-ptx') || has('unsupported-intrinsic') || has('dynamic-parallelism')
  out.push({ id: 'compile', state: ptx ? 'todo' : 'pass', title: 'No PTX, NVIDIA-only intrinsics or device-side launches', detail: ptx ? 'At least one file will not compile under hipcc until these are rewritten.' : 'Nothing that blocks compilation.' })
  if (has('texture-reference')) out.push({ id: 'texture', state: 'todo', title: 'Move texture references to texture objects', detail: 'Texture references are deprecated in HIP.' })
  if (uses('cudaMallocManaged') && t.family === 'CDNA') out.push({ id: 'xnack', state: 'todo', title: 'Enable XNACK for managed memory', detail: 'cudaMallocManaged relies on page migration. On Instinct it needs an xnack+ build and HSA_XNACK=1.', command: 'export HSA_XNACK=1' })
  const fp8 = files.some(f => f.uses.some(u => /fp8|8F_E/.test(u.cuda)))
  if (fp8) out.push({ id: 'fp8', state: t.fp8 === 'none' ? 'todo' : t.fp8 === 'fnuz' ? 'todo' : 'info', title: 'FP8 format matches the target', detail: t.fp8 === 'none' ? `${t.id} has no FP8 hardware; keep a bf16 fallback.` : t.fp8 === 'fnuz' ? `${t.id} uses FNUZ FP8. Weights quantized to OCP e4m3 on NVIDIA must be converted (scale by 2).` : `${t.id} uses OCP FP8, the same encoding as NVIDIA.` })
  if (t.rocm === 'community') out.push({ id: 'override', state: 'todo', title: `${t.id} is not on the ROCm support list`, detail: 'Expect to set HSA_OVERRIDE_GFX_VERSION and build some libraries yourself.', command: 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' })
  out.push({ id: 'verify', state: 'info', title: 'Verify the install', detail: 'torch.cuda works on ROCm; torch.version.hip should be set and your gfx target listed.', command: 'rocminfo | grep -m1 gfx && python -c "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"' })
  out.push({ id: 'groups', state: 'info', title: 'GPU device access', detail: 'Your user needs the render and video groups (or pass /dev/kfd and /dev/dri to the container).', command: 'sudo usermod -aG render,video $USER' })
  return out
}

// ---------- Exports ----------

const csvCell = (s: string | number) => { const v = String(s); return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v }

export function worklistCsv(items: WorkItem[], status: (key: string) => string = () => 'open'): string {
  const head = ['severity', 'category', 'file', 'line', 'title', 'detail', 'fix', 'status']
  return [head.join(','), ...items.map(i => [i.severity, i.category, i.file, i.line, i.title, i.detail, i.fix, status(i.key)].map(csvCell).join(','))].join('\n') + '\n'
}

export interface DiffRow { type: 'same' | 'del' | 'add' | 'mod'; a?: number; b?: number; left?: string; right?: string }

/** Line diff via LCS; changed runs are paired into 'mod' rows for side-by-side display. */
export function diffLines(a: string, b: string): DiffRow[] {
  const A = a.split('\n'), B = b.split('\n')
  const n = A.length, m = B.length
  const ops: ('=' | '-' | '+')[] = []
  if (n * m > 6_000_000) {
    for (let i = 0; i < Math.max(n, m); i++) ops.push(i < n && i < m ? (A[i] === B[i] ? '=' : '-') : i < n ? '-' : '+')
    // Positional fallback: emit a paired add after each changed line.
    const rows: DiffRow[] = []
    for (let i = 0; i < Math.max(n, m); i++) rows.push(A[i] === B[i] ? { type: 'same', a: i + 1, b: i + 1, left: A[i], right: B[i] } : { type: i >= n ? 'add' : i >= m ? 'del' : 'mod', a: i < n ? i + 1 : undefined, b: i < m ? i + 1 : undefined, left: A[i], right: B[i] })
    return rows
  }
  const L = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1))
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1])
  let i = 0, j = 0
  while (i < n || j < m) {
    if (i < n && j < m && A[i] === B[j]) { ops.push('='); i++; j++ }
    else if (j < m && (i >= n || L[i][j + 1] >= L[i + 1][j])) { ops.push('+'); j++ }
    else { ops.push('-'); i++ }
  }
  const rows: DiffRow[] = []
  i = 0; j = 0
  for (let k = 0; k < ops.length;) {
    if (ops[k] === '=') { rows.push({ type: 'same', a: ++i, b: ++j, left: A[i - 1], right: B[j - 1] }); k++; continue }
    const dels: number[] = [], adds: number[] = []
    while (k < ops.length && ops[k] !== '=') { if (ops[k] === '-') dels.push(i++); else adds.push(j++); k++ }
    for (let x = 0; x < Math.max(dels.length, adds.length); x++) {
      const d = dels[x], a2 = adds[x]
      if (d !== undefined && a2 !== undefined) rows.push({ type: 'mod', a: d + 1, b: a2 + 1, left: A[d], right: B[a2] })
      else if (d !== undefined) rows.push({ type: 'del', a: d + 1, left: A[d] })
      else rows.push({ type: 'add', b: a2 + 1, right: B[a2] })
    }
  }
  return rows
}

// ---------- History snapshots ----------

export interface ScanSnapshot {
  id: string
  at: number
  label: string
  target: string
  files: number
  lines: number
  coverage: Coverage
  counts: Record<Severity, number>
  score: number
  items: { key: string; severity: Severity; title: string; file: string }[]
}

export function snapshotOf(r: ScanResult, id: string, label: string, at: number, resolved: (k: string) => boolean = () => false): ScanSnapshot {
  const open = r.items.filter(i => !resolved(i.key))
  return {
    id, at, label, target: r.target.id, files: r.files.length, lines: r.files.reduce((s, f) => s + f.lines, 0), coverage: r.coverage,
    counts: Object.fromEntries(SEVERITIES.map(s => [s, open.filter(i => i.severity === s).length])) as Record<Severity, number>,
    score: readiness(r.items, resolved),
    items: open.map(i => ({ key: i.key, severity: i.severity, title: i.title, file: i.file })),
  }
}

export function compareSnapshots(a: ScanSnapshot, b: ScanSnapshot) {
  const ka = new Set(a.items.map(i => i.key)), kb = new Set(b.items.map(i => i.key))
  return { fixed: a.items.filter(i => !kb.has(i.key)), added: b.items.filter(i => !ka.has(i.key)), still: b.items.filter(i => ka.has(i.key)).length, scoreDelta: b.score - a.score }
}
