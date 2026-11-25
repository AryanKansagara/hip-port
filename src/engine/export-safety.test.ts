import test from 'node:test'
import assert from 'node:assert/strict'
import { ingest } from './ingest.ts'
import { cleanPath, exportPath } from './paths.ts'
import { worklistCsv, type WorkItem } from './scan.ts'
import { readZip, writeZip } from './zip.ts'

const bad = (p: string) => /(^|[\\/])\.\.([\\/]|$)|^[\\/]|^[A-Za-z]:|\\/.test(p)
const nasty = ['../../evil.cu', '/etc/k.cu', 'C:\\x\\k.cu', '\\\\srv\\share\\k.cu', 'a/../../k.cu', '..\\..\\k.cu', 'C:D:k.cu', '/C:/k.cu', 'C:..\\k.cu', '..', '', './', 'a\\..\\..\\k.cu']

test('ordinary paths are unchanged', () => {
  for (const p of ['k.cu', 'src/a-b/c.d.cu', 'a..b.cu', '.hidden/k.h', 'naïve/é.cu', 'a b/c.cu', '...cu', 'src/..x/k.h']) assert.equal(cleanPath(p), p)
})

test('drives, leading slashes and inner dots are resolved; climbing is refused', () => {
  for (const [p, want] of [['/src/k.cu', 'src/k.cu'], ['./k.cu', 'k.cu'], ['C:\\p\\k.cu', 'p/k.cu'], ['src\\k.cu', 'src/k.cu'], ['a/../k.cu', 'k.cu'], ['a//b.cu', 'a/b.cu'], ['../k.cu', null], ['a/../../k.cu', null], ['..\\k.cu', null]] as const) assert.equal(cleanPath(p), want, p)
})

test('no exported path leaves its folder, and none is lost to another', async () => {
  const names = (await readZip(writeZip([...nasty, 'k.cu'].map(path => ({ path, text: 'x' }))))).map(e => e.path)
  assert.equal(new Set(names).size, nasty.length + 1)
  for (const n of names) assert.ok(n && !bad(n), n)
  assert.equal(exportPath('a/k.cu', new Set()), 'a/k.cu')
})

test('an import refuses climbing paths without touching the file they would replace', () => {
  const ds = { files: [{ path: 'evil.cu', text: 'mine' }], deps: null }
  const r = ingest(ds, ['repo/a.cu', '../../evil.cu', 'repo/b.cu'].map(path => ({ path, data: new TextEncoder().encode('int x;') })), true)
  assert.deepEqual(r.ds.files.map(f => [f.path, f.text]), [['a.cu', 'int x;'], ['b.cu', 'int x;'], ['evil.cu', 'mine']])
  assert.equal(r.rows.find(x => x.path === '../../evil.cu')!.status, 'skipped')
})

test('CSV cells cannot start a formula; names and numbers stay', () => {
  const item = (file: string, line = 1): WorkItem => ({ key: 'k', severity: 'low', category: 'api', file, line, title: 't', detail: '', fix: '' })
  const cell = (file: string) => worklistCsv([item(file)]).split('\n').slice(1).join('\n').replace(/^low,api,/, '').replace(/,1,t,,,open\n$/, '')
  for (const c of ['=1+1', '+cmd', '-x.cu', '@SUM(A1)', '\tx', '-1+2']) assert.equal(cell(c), `'${c}`)
  assert.equal(cell('=HYPERLINK("http:evil","x").cu'), `"'=HYPERLINK(""http:evil"",""x"").cu"`)
  assert.equal(cell('\r=1'), `"'\r=1"`)
  for (const c of ['k.cu', 'a=b.cu', '-5', '-1.5', '']) assert.equal(cell(c), c)
  assert.match(worklistCsv([item('k.cu', -3)]), /k\.cu,-3,/)
})
