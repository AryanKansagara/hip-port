// Minimal zip reader and writer. Reading uses the central directory and the platform
// DecompressionStream('deflate-raw'); writing stores files uncompressed with CRC-32.

export interface ZipEntry { path: string; data: Uint8Array }

const u16 = (b: Uint8Array, o: number) => b[o] | (b[o + 1] << 8)
const u32 = (b: Uint8Array, o: number) => (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0

/** Inflates at most `limit` bytes, so a small entry cannot expand without end. */
async function inflateRaw(data: Uint8Array, limit: number): Promise<Uint8Array> {
  const reader = new Blob([data as BlobPart]).stream().pipeThrough(new DecompressionStream('deflate-raw')).getReader()
  const out = new Uint8Array(limit)
  let n = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (n + value.length > limit) { void reader.cancel(); throw new Error('larger than declared') }
    out.set(value, n)
    n += value.length
  }
  return out.subarray(0, n)
}

/**
 * Reads the entries `filter` accepts. Every one is checked against the size and CRC-32 the archive
 * records; an encrypted, damaged or unreadable entry rejects the archive by name. Entries the filter
 * refuses are never decompressed and are passed to `skipped`.
 */
export async function readZip(buf: Uint8Array, filter: (path: string, size: number) => boolean = () => true, skipped: (path: string, size: number) => void = () => {}): Promise<ZipEntry[]> {
  // End of central directory record: scan back over a possible comment.
  let eocd = -1
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 22 - 65535); i--) {
    if (u32(buf, i) === 0x06054b50) { eocd = i; break }
  }
  if (eocd < 0) throw new Error('Not a zip file (no end-of-central-directory record).')
  const count = u16(buf, eocd + 10)
  let p = u32(buf, eocd + 16)
  if (p === 0xffffffff) throw new Error('Zip64 archives are not supported. Re-zip without the large-file option.')
  const dec = new TextDecoder()
  const out: ZipEntry[] = []
  for (let n = 0; n < count; n++) {
    if (p + 46 > buf.length || u32(buf, p) !== 0x02014b50) throw new Error('Corrupt zip central directory.')
    const flags = u16(buf, p + 8)
    const method = u16(buf, p + 10)
    const crc = u32(buf, p + 16)
    const csize = u32(buf, p + 20), size = u32(buf, p + 24)
    const nameLen = u16(buf, p + 28), extraLen = u16(buf, p + 30), commentLen = u16(buf, p + 32)
    const local = u32(buf, p + 42)
    const path = dec.decode(buf.subarray(p + 46, p + 46 + nameLen))
    p += 46 + nameLen + extraLen + commentLen
    if (path.endsWith('/')) continue
    if (!filter(path, size)) { skipped(path, size); continue }
    if (flags & 1) throw new Error(`${path} is encrypted. Zip the sources without a password.`)
    if (method !== 0 && method !== 8) throw new Error(`${path}: unsupported compression method ${method}.`)
    const damaged = new Error(`${path} is damaged in this zip. Zip the sources again.`)
    if (local + 30 > buf.length || u32(buf, local) !== 0x04034b50) throw damaged
    const start = local + 30 + u16(buf, local + 26) + u16(buf, local + 28)
    if (start + csize > buf.length || (method === 0 && csize !== size)) throw damaged
    const raw = buf.subarray(start, start + csize)
    let data: Uint8Array
    try { data = method === 0 ? raw.slice() : await inflateRaw(raw, size) } catch { throw damaged }
    if (data.length !== size || crc32(data) !== crc) throw damaged
    out.push({ path, data })
  }
  return out
}

let crcTable: Uint32Array | null = null
export function crc32(b: Uint8Array): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256)
    for (let i = 0; i < 256; i++) { let c = i; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crcTable[i] = c >>> 0 }
  }
  let c = 0xffffffff
  for (let i = 0; i < b.length; i++) c = crcTable[(c ^ b[i]) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

export function writeZip(files: { path: string; text: string }[]): Uint8Array {
  const enc = new TextEncoder()
  const parts: Uint8Array[] = []
  const central: Uint8Array[] = []
  let offset = 0
  for (const f of files) {
    const name = enc.encode(f.path)
    const data = enc.encode(f.text)
    const crc = crc32(data)
    const h = new DataView(new ArrayBuffer(30))
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint16(8, 0, true)
    h.setUint32(14, crc, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true); h.setUint16(26, name.length, true)
    const c = new DataView(new ArrayBuffer(46))
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true)
    c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, data.length, true); c.setUint16(28, name.length, true); c.setUint32(42, offset, true)
    parts.push(new Uint8Array(h.buffer), name, data)
    central.push(new Uint8Array(c.buffer), name)
    offset += 30 + name.length + data.length
  }
  const csize = central.reduce((s, x) => s + x.length, 0)
  const e = new DataView(new ArrayBuffer(22))
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true); e.setUint32(12, csize, true); e.setUint32(16, offset, true)
  const all = [...parts, ...central, new Uint8Array(e.buffer)]
  const out = new Uint8Array(all.reduce((s, x) => s + x.length, 0))
  let o = 0
  for (const x of all) { out.set(x, o); o += x.length }
  return out
}
