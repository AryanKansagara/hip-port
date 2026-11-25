import test from 'node:test'
import assert from 'node:assert/strict'
import { parseCondaEnv, parseRequirements } from './pydeps.ts'
import { scan } from './scan.ts'

const run = (path: string, text: string) => scan([], { path, text }, { target: 'gfx942', ggl: false })
const names = (text: string) => parseCondaEnv(text).deps.map(d => d.name + d.spec)

test('a file that cannot be read passes no dependency check', () => {
  for (const [path, text] of [
    ['requirements.txt', 'Please sign in to continue\n'], ['requirements.txt', '<!doctype html>\n<title>Sign in</title>\n'], ['requirements.txt', '{"error": "not found"}\n'],
    ['requirements.txt', '# nothing here\n'], ['requirements.txt', 'torch==2.3.1\n404: Not Found\n'], ['environment.yml', 'name: x\n'], ['environment.yml', '<html>\n'], ['environment.yml', 'dependencies:\n'], ['environment.yml', 'dependencies: [a, {pip: [b]}]\n'],
  ]) {
    const r = run(path, text)
    assert.deepEqual(r.checklist.filter(c => ['torch', 'cuda-only', 'forks'].includes(c.id) && c.state === 'pass'), [], text)
    assert.equal(r.items.filter(i => i.key.endsWith('|unread')).length, 1, text)
  }
})

test('a readable file still passes, with no extra item', () => {
  for (const [path, text] of [['requirements.txt', 'torch==2.3.1\nnumpy>=1,<2\n'], ['environment.yml', 'name: x\ndependencies:\n  - numpy=1.26\n']]) {
    const r = run(path, text)
    assert.ok(r.checklist.filter(c => c.id === 'cuda-only' || c.id === 'forks').every(c => c.state === 'pass'))
    assert.ok(!r.items.some(i => i.key.endsWith('|unread')))
  }
})

test('every specifier form pip accepts is still read', () => {
  const lines = ['a', 'a==1', 'a===1-x', 'a~=1.0', 'a!=1.*,>=0', 'a >= 1 , < 2', 'a (>=1,<2)', 'a[x,y]>=1', 'a>=1,', 'a==1!2.0+cu121', 'a==1;m', 'a ; m', 'a=1.2=py3_0', 'a 2.3.1+cu121', 'a==1.0.*', 'a @ https://h/a-1.whl', 'a==1 --hash=sha256:x']
  const r = parseRequirements(lines.join('\n'))
  assert.deepEqual(r.errors, [])
  assert.equal(r.deps.length, lines.length)
})

test('words after a name are not a version', () => {
  const r = parseRequirements('Please sign in\nname: x\nfoo bar\nimport torch\nfoo >= \n')
  assert.deepEqual(r.deps, [])
  assert.deepEqual(r.errors.map(e => e.line), [1, 2, 3, 4, 5])
  const mangled = parseRequirements('torch==2.3.1 +cu121')
  assert.deepEqual([mangled.deps[0].cudaLocal, mangled.errors.length], ['cu121', 1])
})

test('quoted and flow entries are read', () => {
  assert.deepEqual(names('channels: [nvidia, "conda-forge"]\ndependencies: [python=3.11, "pytorch-cuda=12.1", \'numpy>=1,<2\']\n'), ['python=3.11', 'pytorch-cuda=12.1', 'numpy>=1,<2'])
  assert.deepEqual(parseCondaEnv('channels: [nvidia]\ndependencies: [a]').cudaChannels, ['nvidia'])
  assert.deepEqual(names('dependencies: [\n  a,\n  b=2\n]\nname: x\n'), ['a', 'b=2'])
  assert.deepEqual(names('dependencies:\n- "a>=1"\n- pip: [b==1, "c==2"]\n- d\n'), ['a>=1', 'b==1', 'c==2', 'd'])
  assert.deepEqual(names('dependencies:\n  - pip\n  - pip:\n    - "torch==2.3.1+cu121"\n  - _libgcc_mutex=0.1=main\n'), ['pip', 'torch==2.3.1+cu121', '_libgcc_mutex=0.1=main'])
  for (const t of ['dependencies:\n  - a\n', 'dependencies: []\n', '---\nname: x\ndependencies:\n  - a  # c\nprefix: /x\nvariables:\n  A: b\n']) assert.deepEqual(parseCondaEnv(t).errors, [], t)
})

test('problems inside pip: and in the list are reported on their line', () => {
  const r = parseCondaEnv('dependencies:\n  - numpy\n  - pip:\n    - -r base.txt\n    - oh no\n  - please sign in\n  what\n')
  assert.deepEqual(r.errors.map(e => e.line), [4, 5, 6, 7])
  assert.equal(parseCondaEnv('dependencies: [a, b\n').errors.length, 1)
})
