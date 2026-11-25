import test from 'node:test'
import assert from 'node:assert/strict'
import { deflateRawSync, crc32 } from 'node:zlib'
import { readZip, writeZip } from './zip.ts'

interface F { path: string; data: string; stored?: boolean; method?: number; flags?: number; crc?: number; size?: number; local?: number }
function zip(files: F[]): Uint8Array {
  const parts: Buffer[] = [], dir: Buffer[] = []
  let off = 0
  for (const f of files) {
    const raw = Buffer.from(f.data), body = f.stored ? raw : deflateRawSync(raw), name = Buffer.from(f.path)
    const h = Buffer.alloc(26)
    h.writeUInt16LE(20, 0); h.writeUInt16LE(f.flags ?? 0, 2); h.writeUInt16LE(f.method ?? (f.stored ? 0 : 8), 4)
    h.writeUInt32LE(f.crc ?? crc32(raw), 10); h.writeUInt32LE(body.length, 14); h.writeUInt32LE(f.size ?? raw.length, 18); h.writeUInt16LE(name.length, 22)
    const at = Buffer.alloc(4); at.writeUInt32LE(f.local ?? off)
    const l = Buffer.concat([Buffer.from('PK\x03\x04'), h, name, body])
    parts.push(l); off += l.length
    dir.push(Buffer.concat([Buffer.from('PK\x01\x02'), Buffer.from([20, 0]), h, Buffer.alloc(10), at, name]))
  }
  const d = Buffer.concat(dir), e = Buffer.alloc(18)
  e.writeUInt16LE(files.length, 4); e.writeUInt16LE(files.length, 6); e.writeUInt32LE(d.length, 8); e.writeUInt32LE(off, 12)
  return new Uint8Array(Buffer.concat([...parts, d, Buffer.from('PK\x05\x06'), e]))
}
const CU = 'cudaMalloc(&p, 4);\n'.repeat(20)
const ok = { path: 'r/ok.cu', data: 'int x;\n' }

test('a flipped byte, a wrong size, a bad offset or a cut deflate stream is damage, by name', async () => {
  const flipped = zip([{ path: 'r/k.cu', data: CU, stored: true }]); flipped[60] ^= 1
  const cut = zip([{ path: 'r/k.cu', data: CU }]); cut.fill(0xff, 40, 50)
  for (const z of [flipped, cut, zip([ok, { path: 'r/k.cu', data: CU, crc: 1 }]), zip([{ path: 'r/k.cu', data: CU, size: 5 }]), zip([{ path: 'r/k.cu', data: CU, size: 5000 }]), zip([{ path: 'r/k.cu', data: CU, stored: true, size: 5 }]), zip([{ path: 'r/k.cu', data: CU, local: 99999 }]), zip([ok, { path: 'r/k.cu', data: CU, local: 3 }])])
    await assert.rejects(readZip(z), /^Error: r\/k\.cu is damaged/)
})

test('encrypted entries are refused by name', async () => {
  await assert.rejects(readZip(zip([ok, { path: 'r/k.cu', data: CU, stored: true, flags: 1 }])), /r\/k\.cu is encrypted/)
})

test('entries the filter refuses are listed and never decompressed', async () => {
  const z = zip([ok, { path: 'r/w.bin', data: 'x', stored: true, method: 93 }, { path: 'r/e.bin', data: 'x', flags: 1 }, { path: 'r/bad.bin', data: 'abc', crc: 1 }, { path: 'r/dir/', data: '', stored: true }])
  const skipped: [string, number][] = []
  const got = await readZip(z, p => p.endsWith('.cu'), (p, n) => skipped.push([p, n]))
  assert.deepEqual(got.map(e => e.path), ['r/ok.cu'])
  assert.deepEqual(skipped, [['r/w.bin', 1], ['r/e.bin', 1], ['r/bad.bin', 3]])
})

test('good archives read as before: stored, deflated, empty, UTF-8 and data-descriptor flags', async () => {
  const z = zip([{ path: 'a.cu', data: CU }, { path: 'b.cu', data: CU, stored: true }, { path: 'c.cu', data: '', stored: true }, { path: 'd.cu', data: '' }, { path: 'é.cu', data: 'é', flags: 0x0808 }])
  assert.deepEqual((await readZip(z)).map(e => [e.path, new TextDecoder().decode(e.data)]), [['a.cu', CU], ['b.cu', CU], ['c.cu', ''], ['d.cu', ''], ['é.cu', 'é']])
  assert.equal((await readZip(writeZip([{ path: 'x', text: CU }])))[0].data.length, CU.length)
  assert.deepEqual(await readZip(zip([])), [])
})

test('a directory that runs past the end of the file is corrupt', async () => {
  const z = zip([ok]); new DataView(z.buffer).setUint16(z.length - 12, 3, true)
  await assert.rejects(readZip(z), /Corrupt zip central directory/)
})
