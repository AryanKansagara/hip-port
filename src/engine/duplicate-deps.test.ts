import test from 'node:test'
import assert from 'node:assert/strict'
import { readiness, scan } from './scan.ts'

const run = (path: string, text: string) => scan([], { path, text }, { target: 'gfx942', ggl: false })
const torchCheck = (path: string, text: string) => run(path, text).checklist.find(c => c.id === 'torch')!

test('the worst entry of a package is kept, whatever the order of the lines', () => {
  const lines = ['torch==2.1.0 ; sys_platform == "darwin"', 'torch==2.3.1+cu121 ; sys_platform == "linux"']
  for (const ls of [lines, [...lines].reverse()]) {
    const r = run('requirements.txt', ls.join('\n'))
    assert.deepEqual(r.items.map(i => [i.severity, i.line]), [['high', ls.indexOf(lines[1]) + 1]])
    assert.equal(readiness(r.items), 95)
    assert.equal(r.counts.high, 1)
    const c = r.checklist.find(c => c.id === 'torch')!
    assert.equal(c.state, 'todo')
    assert.match(c.detail, new RegExp(`Line ${ls.indexOf(lines[1]) + 1} pins a CUDA build \\(\\+cu121\\)`))
  }
})

test('the torch check looks at every torch entry', () => {
  assert.equal(torchCheck('requirements.txt', 'torch==2.1.0 ; a\ntorch==2.2.0 ; b\n').state, 'pass')
  assert.equal(torchCheck('requirements.txt', 'torch==2.1.0\nTorch==2.2.0+cu118\n').state, 'todo')
  for (const [a, b] of [['pytorch=2.3', 'numpy'], ['numpy', 'pytorch=2.3']]) assert.equal(torchCheck('environment.yml', `dependencies:\n  - ${a}\n  - ${b}\n  - pip:\n    - torch==2.1.0\n`).state, 'todo')
})

test('one row per package, and other packages are unaffected', () => {
  const r = run('requirements.txt', 'vllm\ntorch\nvllm\nnumpy\n')
  assert.equal(r.items.filter(i => i.title.startsWith('vllm')).length, 1)
  assert.equal(r.items.find(i => i.title.startsWith('vllm'))!.line, 1)
})
