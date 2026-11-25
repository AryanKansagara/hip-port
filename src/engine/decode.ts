// Source bytes to text. Exports are UTF-8, so a byte read wrongly here is a character changed there.

/** The text and, when the bytes were not plain UTF-8, a remark; or only the reason the file cannot be read. */
export interface Decoded { text?: string; remark?: string }

const strict = (enc: string, b: Uint8Array) => { try { return new TextDecoder(enc, { fatal: true }).decode(b) } catch { return undefined } }

export function decodeSource(b: Uint8Array): Decoded {
  const utf16 = b[0] === 0xff && b[1] === 0xfe ? 'LE' : b[0] === 0xfe && b[1] === 0xff ? 'BE' : null
  // FF FE 00 00 also starts UTF-32; its NULs are caught below.
  if (utf16) {
    const text = strict(`utf-16${utf16.toLowerCase()}`, b)
    if (text === undefined) return { remark: `Damaged UTF-16 ${utf16}; save it again as UTF-8` }
    if (text.slice(0, 4096).includes('\0')) return { remark: 'Binary content' }
    return { text, remark: `Read as UTF-16 ${utf16}; exports are UTF-8` }
  }
  if (b.subarray(0, 4096).includes(0)) return { remark: 'Binary content' }
  const text = strict('utf-8', b)
  if (text !== undefined) return { text }
  // Mostly valid multi-byte sequences: UTF-8 that was cut or corrupted, where Windows-1252 would garble the good part.
  const { good, bad, line } = utf8Sequences(b)
  if (good > bad) return { remark: `Damaged UTF-8 on line ${line}; save it again as UTF-8` }
  return { text: new TextDecoder('windows-1252').decode(b), remark: 'Not UTF-8: read as Windows-1252; exports are UTF-8. Check accented characters' }
}

/** Counts well-formed multi-byte sequences and bytes that fit none, with the line of the first such byte. */
function utf8Sequences(b: Uint8Array) {
  let good = 0, bad = 0, line = 1, at = 0
  for (let i = 0; i < b.length;) {
    const c = b[i]
    if (c < 0x80) { if (c === 10 && !at) line++; i++; continue }
    const n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0
    if (n && i + n <= b.length && strict('utf-8', b.subarray(i, i + n)) !== undefined) { good++; i += n }
    else { bad++; at ||= 1; i++ }
  }
  return { good, bad, line }
}
