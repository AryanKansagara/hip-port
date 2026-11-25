import test from 'node:test'
import assert from 'node:assert/strict'
import { ingest, type Incoming } from './ingest.ts'
import type { Dataset } from '../store.ts'

const none: Dataset = { files: [], deps: null }
const file = (path: string, text = 'int x;\n'): Incoming => ({ path, data: new TextEncoder().encode(text) })
const paths = (ds: Dataset) => [...ds.files.map(f => f.path), ...(ds.deps ? [ds.deps.path] : [])]

test('a pasted path is kept as typed', () => {
  for (const p of ['src/util.h', 'a/b/c.cu', 'build/gen.cu', 'k.cu', 'env/requirements.txt']) assert.deepEqual(paths(ingest(none, [file(p)], false).ds), [p])
  const root: Dataset = { files: [{ path: 'util.h', text: 'root' }], deps: null }
  assert.deepEqual(ingest(root, [file('src/util.h', 'nested')], false).ds.files, [{ path: 'src/util.h', text: 'nested' }, { path: 'util.h', text: 'root' }])
})

test('a wrapper folder is dropped; a folder that is part of the layout is not', () => {
  const after = (ds: Dataset, ...ps: string[]) => paths(ingest(ds, ps.map(p => file(p)), true).ds)
  assert.deepEqual(after(none, 'repo-main/src/k.cu', 'repo-main/u.h'), ['src/k.cu', 'u.h'])
  assert.deepEqual(after(none, 'repo/src/k.cu'), ['src/k.cu'])
  assert.deepEqual(after(none, 'src/k.cu', 'src/u.h'), ['src/k.cu', 'src/u.h'])
  assert.deepEqual(after(none, 'csrc/k.cu'), ['csrc/k.cu'])
  assert.deepEqual(after(none, 'a/k.cu', 'b/k.cu'), ['a/k.cu', 'b/k.cu'])
  assert.deepEqual(after(none, 'a/k.cu', 'k.cu'), ['a/k.cu', 'k.cu'])
  // The same word as a prefix of another folder is not a shared folder.
  assert.deepEqual(after(none, 'a/k.cu', 'ab/k.cu'), ['a/k.cu', 'ab/k.cu'])
  assert.deepEqual(after({ files: [{ path: 'solver/k.cu', text: '' }], deps: null }, 'solver/u.h'), ['solver/k.cu', 'solver/u.h'])
  assert.deepEqual(ingest(none, [], true).rows, [])
})

test('replacing a file counts the difference; the limit still applies to new content', () => {
  const files = (n: number, text: string) => Array.from({ length: n }, (_, i) => file(`f${i}.cu`, text))
  const big = 'x'.repeat(900_000)
  const first = ingest(none, files(27, big), false) // 24.3 MB
  assert.ok(first.rows.every(r => r.status === 'added'))
  assert.ok(ingest(first.ds, files(27, big), false).rows.every(r => r.status === 'replaced'))
  assert.ok(ingest(first.ds, files(27, 'y'), false).rows.every(r => r.status === 'replaced'))
  assert.match(ingest(first.ds, [file('new.cu', big)], false).rows[0].reason!, /limit/)
  // A replacement that grows past the limit keeps the old copy.
  const grown = ingest(first.ds, files(27, 'x'.repeat(999_999)), false)
  assert.equal(grown.rows.filter(r => r.status === 'replaced').length, 7)
  assert.ok(grown.ds.files.reduce((s, f) => s + f.text.length, 0) <= 25_000_000)
  // The same path twice in one import is one file.
  assert.equal(ingest(first.ds, [file('n.cu', 'x'.repeat(600_000)), file('n.cu', 'x'.repeat(600_000))], false).rows[1].status, 'replaced')
  // Dependency files count too, and replacing one counts the difference.
  const dep = ingest(first.ds, [file('requirements.txt', '#'.repeat(690_000))], false)
  assert.equal(ingest(dep.ds, [file('environment.yml', '#'.repeat(690_000))], false).rows[0].status, 'replaced')
  assert.equal(ingest(dep.ds, [file('new.cu', 'x'.repeat(20_000))], false).rows[0].status, 'skipped')
  // Stored size counts, not bytes: é is two bytes and one character.
  assert.equal(ingest(first.ds, [file('é.cu', 'é'.repeat(450_000))], false).rows[0].status, 'added')
})
