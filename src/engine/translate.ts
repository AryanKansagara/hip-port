// CUDA -> HIP source translation for one file.
import { mask, lineOf, matchBracket, splitArgs, type Masked } from './lex.ts'
import { lookup, lookupHeader, type Lib, type Mapping } from './mapping.ts'

export interface Use {
  cuda: string
  hip: string | null
  lib: Lib
  status: Mapping['status'] | 'unmapped'
  via: Mapping['via'] | 'none'
  note?: string
  kind: 'api' | 'header' | 'launch'
  lines: number[]
}

export interface Translation {
  out: string
  uses: Use[]
  /** Number of <<< >>> launches found (and converted when asked) */
  launches: number
  addedInclude: boolean
}

export interface TranslateOptions {
  /** Rewrite kernel<<<g, b, s, st>>>(args) as hipLaunchKernelGGL(kernel, ...) */
  ggl?: boolean
  /** Names defined by the project itself (macros, functions) that must never be renamed */
  defined?: Set<string>
}

interface Edit { start: number; end: number; text: string }

// Anything that looks like a CUDA API name but has no mapping is reported as unmapped.
const CUDA_LIKE = /^(?:cuda[A-Z_]|cu[A-Z]|CU_|CUDA_|CUDART_|cublas|CUBLAS|cufft|CUFFT|curand|CURAND|cusparse|CUSPARSE|cusolver|CUSOLVER|cudnn|CUDNN|nvtx|nccl)/

/** Macro names and function names defined in this source, so a user's own cudaCheck() survives. */
export function definedNames(src: string, m: Masked = mask(src)): Set<string> {
  const s = new Set<string>()
  for (const x of m.code.matchAll(/#\s*define\s+(\w+)/g)) s.add(x[1])
  for (const x of m.code.matchAll(/^[ \t]*(?:(?:static|inline|__host__|__device__|__forceinline__|template\s*<[^>]*>)\s+)*[\w:<>*&]+[\s*&]+(\w+)\s*\([^;{]*\)\s*\{/gm)) s.add(x[1])
  return s
}

export function translate(src: string, opts: TranslateOptions = {}): Translation {
  const m = mask(src)
  const defined = opts.defined ?? definedNames(src, m)
  const edits: Edit[] = []
  const uses = new Map<string, Use>()
  const note = (u: Omit<Use, 'lines'>, line: number) => {
    const k = u.kind + ':' + u.cuda
    const e = uses.get(k)
    if (e) { if (!e.lines.includes(line)) e.lines.push(line) }
    else uses.set(k, { ...u, lines: [line] })
  }

  // Headers
  const includeSpans: [number, number][] = []
  let hasHipRuntime = false
  for (const x of m.code.matchAll(/^[ \t]*#[ \t]*include[ \t]*([<"])([^>"\n]+)[>"]/gm)) {
    const lineEnd = x.index! + x[0].length
    includeSpans.push([x.index!, lineEnd])
    const name = x[2].trim()
    const start = x.index! + x[0].length - 1 - x[2].length
    if (name.startsWith('hip/hip_runtime')) hasHipRuntime = true
    const h = lookupHeader(name)
    if (!h) continue
    const line = lineOf(m, x.index!)
    note({ cuda: name, hip: h.hip, lib: h.lib, status: h.status, via: h.via, note: h.note, kind: 'header' }, line)
    if (h.hip && h.hip !== name) {
      edits.push({ start, end: start + x[2].length, text: h.hip })
      if (h.hip.startsWith('hip/hip_runtime')) hasHipRuntime = true
    }
  }
  const inInclude = (i: number) => includeSpans.some(([a, b]) => i >= a && i < b)

  // Identifiers
  let usesCuda = false
  for (const x of m.code.matchAll(/[A-Za-z_]\w*/g)) {
    const id = x[0], at = x.index!
    if (at > 0 && /[\w.]/.test(m.code[at - 1])) continue
    if (m.code.slice(Math.max(0, at - 2), at) === '->') continue
    if (/^\d/.test(id) || inInclude(at)) continue
    if (id === 'cub' && /^\s*::/.test(m.code.slice(at + 3, at + 8))) {
      edits.push({ start: at, end: at + 3, text: 'hipcub' })
      note({ cuda: 'cub::', hip: 'hipcub::', lib: 'cub', status: 'ok', via: 'table', kind: 'api' }, lineOf(m, at))
      usesCuda = true
      continue
    }
    if (id === 'thrust' && /^\s*::/.test(m.code.slice(at + 6, at + 11))) {
      note({ cuda: 'thrust::', hip: 'thrust::', lib: 'thrust', status: 'ok', via: 'table', note: 'rocThrust keeps the thrust namespace.', kind: 'api' }, lineOf(m, at))
      continue
    }
    if (defined.has(id)) continue
    const mp = lookup(id)
    if (!mp) {
      if (CUDA_LIKE.test(id)) note({ cuda: id, hip: null, lib: 'runtime', status: 'unmapped', via: 'none', note: 'Looks like a CUDA API name but is not in the mapping table.', kind: 'api' }, lineOf(m, at))
      continue
    }
    usesCuda = true
    note({ cuda: id, hip: mp.hip, lib: mp.lib, status: mp.status, via: mp.via, note: mp.note, kind: 'api' }, lineOf(m, at))
    if (mp.hip && mp.hip !== id) edits.push({ start: at, end: at + id.length, text: mp.hip })
  }

  // Kernel launches
  const launches = findLaunches(m)
  for (const l of launches) {
    note({ cuda: '<<<…>>>', hip: opts.ggl ? 'hipLaunchKernelGGL' : '<<<…>>>', lib: 'runtime', status: 'ok', via: 'table', note: 'hipcc accepts triple-chevron launches as is.', kind: 'launch' }, lineOf(m, l.start))
    if (!opts.ggl) continue
    const piece = (a: number, b: number) => applyEdits(src.slice(a, b), edits.filter(e => e.start >= a && e.end <= b).map(e => ({ ...e, start: e.start - a, end: e.end - a }))).trim()
    const name = piece(l.nameStart, l.nameEnd)
    const cfg = l.config.map(([a, b]) => piece(a, b))
    const args = l.argsEnd > l.argsStart ? piece(l.argsStart, l.argsEnd) : ''
    const kname = /[<>:]/.test(name) ? `(${name})` : name
    const text = `hipLaunchKernelGGL(${kname}, dim3(${cfg[0]}), dim3(${cfg[1]}), ${cfg[2] ?? '0'}, ${cfg[3] ?? '0'}${args ? ', ' + args : ''})`
    // Drop identifier edits inside the launch; they are already applied in `text`.
    for (let i = edits.length - 1; i >= 0; i--) if (edits[i].start >= l.nameStart && edits[i].end <= l.end) edits.splice(i, 1)
    edits.push({ start: l.nameStart, end: l.end, text })
  }

  let out = applyEdits(src, edits)
  let addedInclude = false
  const hasKernels = /\b__global__\b|<<</.test(m.code)
  if (!hasHipRuntime && (usesCuda || hasKernels)) {
    // Put it after any leading comment block / pragma once so the diff stays readable.
    const lead = src.match(/^(?:\s*(?:\/\/[^\n]*|\/\*[\s\S]*?\*\/|#\s*pragma\s+once[^\n]*)\n)*/)![0]
    out = out.slice(0, lead.length) + '#include <hip/hip_runtime.h>\n' + out.slice(lead.length)
    addedInclude = true
  }
  return { out, uses: [...uses.values()], launches: launches.length, addedInclude }
}

function applyEdits(s: string, edits: Edit[]): string {
  const sorted = [...edits].sort((a, b) => b.start - a.start)
  let out = s
  let floor = Infinity
  for (const e of sorted) {
    if (e.end > floor) continue // overlapping edit; keep the later (outer) one
    out = out.slice(0, e.start) + e.text + out.slice(e.end)
    floor = e.start
  }
  return out
}

export interface Launch { start: number; nameStart: number; nameEnd: number; config: [number, number][]; argsStart: number; argsEnd: number; end: number }

/** Finds name<<<cfg>>>(args) launches in masked code. */
export function findLaunches(m: Masked): Launch[] {
  const code = m.code
  const out: Launch[] = []
  let from = 0
  for (;;) {
    const open = code.indexOf('<<<', from)
    if (open < 0) break
    from = open + 3
    const close = code.indexOf('>>>', open + 3)
    if (close < 0) break
    // Kernel name: identifier, optional template args and namespaces, scanning backwards.
    let i = open
    while (i > 0 && /\s/.test(code[i - 1])) i--
    const nameEnd = i
    if (code[i - 1] === '>') {
      let depth = 0
      for (i = i - 1; i >= 0; i--) {
        if (code[i] === '>') depth++
        else if (code[i] === '<' && --depth === 0) break
      }
    }
    while (i > 0 && /[\w:]/.test(code[i - 1])) i--
    const nameStart = i
    if (nameStart === nameEnd) continue
    const cfgInner = code.slice(open + 3, close)
    const config = splitArgs(cfgInner).map(a => [open + 3 + a.start, open + 3 + a.start + a.text.length] as [number, number])
    let p = close + 3
    while (p < code.length && /\s/.test(code[p])) p++
    if (code[p] !== '(') continue
    const pe = matchBracket(code, p)
    if (pe < 0) continue
    out.push({ start: open, nameStart, nameEnd, config, argsStart: p + 1, argsEnd: pe, end: pe + 1 })
    from = pe + 1
  }
  return out
}

/** foo.cu -> foo.hip; headers keep their names so includes still resolve. */
export const hipPath = (p: string) => p.replace(/\.cu$/i, '.hip')
