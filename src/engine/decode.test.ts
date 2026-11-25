import test from 'node:test'
import assert from 'node:assert/strict'
import { decodeSource } from './decode.ts'

const le = (s: string) => new Uint8Array(Buffer.from('﻿' + s, 'utf16le'))
const be = (s: string) => { const b = Buffer.from('﻿' + s, 'utf16le'); return new Uint8Array(b.swap16()) }
const latin1 = (s: string) => new Uint8Array(Buffer.from(s, 'latin1'))
const utf8 = (s: string) => new TextEncoder().encode(s)

test('valid UTF-8 is unchanged and unremarked, with or without a byte order mark', () => {
  for (const s of ['int x;\n', '// José 日本 🎉\r\n', '', '﻿int y;', 'a\fb']) assert.deepEqual(decodeSource(utf8(s)), { text: s.replace('﻿', '') })
})

test('other bytes are read as Windows-1252 with a remark', () => {
  for (const [b, want] of [[latin1('// José\n'), '// José\n'], [new Uint8Array([0x93, 0x78, 0x94, 0x80]), '“x”€'], [latin1('«É» père'), '«É» père'], [latin1('é'), 'é']] as const) {
    const d = decodeSource(b)
    assert.equal(d.text, want)
    assert.match(d.remark!, /Windows-1252/)
  }
})

test('UTF-16 with a byte order mark is read, and refused when damaged', () => {
  assert.equal(decodeSource(le('torch==2.3.1\r\nJosé 🎉')).text, 'torch==2.3.1\r\nJosé 🎉')
  assert.equal(decodeSource(be('torch')).text, 'torch')
  assert.match(decodeSource(be('torch')).remark!, /UTF-16 BE/)
  for (const b of [le('torch').subarray(0, 7), new Uint8Array([0xff, 0xfe, 0x00, 0xd8, 0x41, 0x00])]) {
    const d = decodeSource(b)
    assert.equal(d.text, undefined)
    assert.match(d.remark!, /Damaged UTF-16/)
  }
})

test('binary is still skipped: NULs, UTF-32, UTF-16 without a mark', () => {
  for (const b of [new Uint8Array([1, 0, 2]), new Uint8Array([0xff, 0xfe, 0, 0, 0x41, 0, 0, 0]), new Uint8Array(Buffer.from('torch', 'utf16le'))]) assert.deepEqual(decodeSource(b), { remark: 'Binary content' })
})

test('UTF-8 that was cut or corrupted is refused rather than garbled', () => {
  const s = utf8('// é\n// ü ñ\nx = "日"')
  assert.deepEqual(decodeSource(s.subarray(0, s.length - 2)), { remark: 'Damaged UTF-8 on line 3; save it again as UTF-8' })
})
