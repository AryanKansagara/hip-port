// A small C/C++/CUDA scanner. It does not build tokens for the whole language; it produces a
// "masked" copy of the source where comments and string/char literals are blanked out (same
// length, newlines kept), so later passes can search code with plain regexes and still map every
// match back to the original offsets.

export interface Masked {
  /** Same length as the source. Comments and literal bodies are spaces. */
  code: string
  /** Offsets where each line starts, for offset -> line lookups. */
  lineStarts: number[]
}

export function mask(src: string): Masked {
  const out = src.split('')
  const n = src.length
  let i = 0
  const blank = (a: number, b: number) => { for (let k = a; k < b; k++) if (out[k] !== '\n') out[k] = ' ' }
  // Track #include lines so `<cuda.h>` and "cublas_v2.h" stay visible as header names.
  let lineStart = 0
  while (i < n) {
    const c = src[i]
    if (c === '\n') { i++; lineStart = i; continue }
    if (c === '/' && src[i + 1] === '/') {
      let j = i
      while (j < n && src[j] !== '\n') {
        // A backslash-newline continues a line comment.
        if (src[j] === '\\' && src[j + 1] === '\n') j++
        j++
      }
      blank(i, j); i = j; continue
    }
    if (c === '/' && src[i + 1] === '*') {
      const end = src.indexOf('*/', i + 2)
      const j = end < 0 ? n : end + 2
      blank(i, j); i = j; continue
    }
    if (c === 'R' && src[i + 1] === '"' && !isIdent(src[i - 1])) {
      // Raw string literal R"delim( ... )delim"
      const open = src.indexOf('(', i + 2)
      if (open > 0 && open - i < 20) {
        const delim = src.slice(i + 2, open)
        const close = src.indexOf(')' + delim + '"', open)
        const j = close < 0 ? n : close + delim.length + 2
        blank(i + 2, j - 1); i = j; continue
      }
    }
    if (c === '"' || c === "'") {
      if (c === "'" && isDigitSep(src, i)) { i++; continue }
      let j = i + 1
      while (j < n && src[j] !== c && src[j] !== '\n') { if (src[j] === '\\') j++; j++ }
      const isInclude = c === '"' && /^\s*#\s*include\s*$/.test(src.slice(lineStart, i))
      if (!isInclude) blank(i + 1, j)
      i = j + 1; continue
    }
    i++
  }
  const lineStarts = [0]
  for (let k = 0; k < n; k++) if (src[k] === '\n') lineStarts.push(k + 1)
  return { code: out.join(''), lineStarts }
}

const isIdent = (c: string | undefined) => !!c && /[A-Za-z0-9_]/.test(c)
// C++14 digit separators: 1'000'000
const isDigitSep = (s: string, i: number) => /[0-9a-fA-F]/.test(s[i - 1] ?? '') && /[0-9a-fA-F]/.test(s[i + 1] ?? '') && /\d/.test(backToNumberStart(s, i))
const backToNumberStart = (s: string, i: number) => { let k = i - 1; while (k > 0 && /[0-9a-fA-FxX']/.test(s[k - 1])) k--; return s[k] }

/** 1-based line number for an offset. */
export function lineOf(m: Masked, offset: number): number {
  let lo = 0, hi = m.lineStarts.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (m.lineStarts[mid] <= offset) lo = mid
    else hi = mid - 1
  }
  return lo + 1
}

/** Index of the bracket matching the one at `open` in masked code, or -1. */
export function matchBracket(code: string, open: number): number {
  const o = code[open]
  const c = o === '(' ? ')' : o === '[' ? ']' : o === '{' ? '}' : '>'
  let depth = 0
  for (let i = open; i < code.length; i++) {
    if (code[i] === o) depth++
    else if (code[i] === c && --depth === 0) return i
  }
  return -1
}

/** Splits an argument list on top-level commas. `s` is masked code without the outer parens. */
export function splitArgs(s: string): { text: string; start: number }[] {
  const out: { text: string; start: number }[] = []
  let depth = 0, start = 0
  for (let i = 0; i < s.length; i++) {
    const c = s[i]
    if (c === '(' || c === '[' || c === '{') depth++
    else if (c === ')' || c === ']' || c === '}') depth--
    else if (c === ',' && depth === 0) { out.push({ text: s.slice(start, i), start }); start = i + 1 }
  }
  if (s.trim() || out.length) out.push({ text: s.slice(start), start })
  return out
}
