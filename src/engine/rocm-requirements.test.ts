import test from 'node:test'
import assert from 'node:assert/strict'
import { parseCondaEnv, parseRequirements, rocmRequirements } from './pydeps.ts'
import { SAMPLES } from './sample.ts'

const pip = (s: string) => rocmRequirements(parseRequirements(s)).split('\n').slice(2, -1)
const conda = (...l: string[]) => rocmRequirements(parseCondaEnv('dependencies:\n' + l.map(x => `  - ${x}\n`).join(''))).split('\n').slice(2, -1)

test('sources, markers and editable installs are kept', () => {
  for (const l of ['peft @ git+https://github.com/huggingface/peft.git@v0.13.0', 'git+https://github.com/x/y.git@v1#egg=mylib', 'git+ssh://git@github.com/a/noegg.git', '-e git+https://github.com/a/b.git#egg=bb', '-e .', 'foo>=1 ; python_version > "3.8"', 'foo[bar]>=1,<2'])
    assert.deepEqual(pip(l), [l])
  assert.deepEqual(pip('torch==2.1.0 ; sys_platform == "darwin"\ntorch==2.3.1+cu121;sys_platform=="linux"'), ['torch==2.1.0 ; sys_platform == "darwin"', 'torch==2.3.1 ; sys_platform=="linux"'])
  assert.deepEqual(pip('onnxruntime-gpu==1.18 ; sys_platform == "linux"'), ['onnxruntime-rocm ; sys_platform == "linux"'])
})

test('continuations are one requirement and keep their hashes', () => {
  assert.deepEqual(pip('einops==0.8.0 \\\n    --hash=sha256:aa \\\n    --hash=sha256:bb\nnumpy==1.26.4 ; python_version < "3.13" \\\r\n    --hash=sha256:cc\r\nidna==3'), ['einops==0.8.0 --hash=sha256:aa --hash=sha256:bb', 'numpy==1.26.4 ; python_version < "3.13" --hash=sha256:cc', 'idna==3'])
  // The hash of a CUDA wheel never matches a ROCm wheel.
  assert.deepEqual(pip('torch==2.3.1+cu121 \\\n    --hash=sha256:aa\nnumpy'), ['torch==2.3.1', 'numpy'])
  assert.equal(parseRequirements('a \\\n  --hash=x\nb').deps[1].line, 3)
  assert.deepEqual(pip('# note \\\nnumpy'), ['numpy'])
  assert.deepEqual(pip('numpy \\'), ['numpy'])
})

test('other indexes are kept, without credentials', () => {
  assert.deepEqual(pip('--index-url https://user:s3cret@pypi.example.test/simple\n-f https://wheels.example.test/x.html\n--extra-index-url=https://download.pytorch.org/whl/cu121\n-f https://download.pytorch.org/whl/torch_stable.html\nnumpy'), ['--index-url https://pypi.example.test/simple', '--find-links https://wheels.example.test/x.html', 'numpy'])
  assert.deepEqual(pip('x @ git+https://tok:en@github.com/a/b.git'), ['x @ git+https://github.com/a/b.git'])
  assert.ok(!JSON.stringify(parseRequirements('--extra-index-url https://u:s3cret@host/whl/cu121').cudaIndexes).includes('s3cret'))
})

test('CUDA extras go, other extras stay', () => {
  assert.deepEqual(pip('jax[cuda12]==0.4.30\njax[cuda12_local,k8s]\nfoo[cudaless]'), ['jax==0.4.30', 'jax[k8s]', 'foo[cudaless]'])
})

test('conda specs become pip specs', () => {
  assert.deepEqual(conda('python=3.11', 'pytorch=2.3.0=py3.11_cuda12.1_cudnn8.9.2_0', 'pytorch=2.4', 'pytorch', 'numpy 1.26.*', 'scipy=1.11', 'scipy==1.11.4', 'pandas>=2,<3', 'pillow=10.2.0=py311_0', 'tqdm 4.66.1 pyhd8ed1ab_0', 'six=1.*', 'pip'), [
    '# removed python: set by the environment, not installed with pip.', 'torch==2.3.0', 'torch==2.4.*', 'torch', 'numpy==1.26.*', 'scipy==1.11.*', 'scipy==1.11.4', 'pandas>=2,<3', 'pillow==10.2.0', 'tqdm==4.66.1', 'six==1.*', 'pip',
  ])
})

test('conda list exports and indexes inside a pip: list', () => {
  assert.deepEqual(pip('numpy=1.26.4=py311_0'), ['numpy==1.26.4'])
  assert.deepEqual(rocmRequirements(parseCondaEnv('dependencies:\n  - pip:\n    - --extra-index-url https://pypi.example.test/simple\n    - foo==1\n')).split('\n').slice(2, -1), ['--extra-index-url https://pypi.example.test/simple', 'foo==1'])
})

test('conda forms without a pip twin are comments, not broken lines', () => {
  assert.deepEqual(conda('numpy=*', 'a=1.2|1.3', 'b[version=">1"]', 'c=1.26.*=py311_0', 'pytorch=2.3.0+cu121', 'tz=2024a', 'd >=1,<2'), ['numpy', '# check a: the conda spec =1.2|1.3 has no pip form.', '# check b: the conda spec [version=">1"] has no pip form.', 'c==1.26.*', 'torch==2.3.0', 'tz==2024a', 'd>=1,<2'])
})

test('every line of the bundled samples is something pip can parse', () => {
  for (const s of SAMPLES) for (const f of s.files.filter(x => /txt$|yml$/.test(x.path))) {
    const out = rocmRequirements(f.path.endsWith('.txt') ? parseRequirements(f.text) : parseCondaEnv(f.text))
    for (const l of out.trim().split('\n')) assert.match(l, /^(#.*|--extra-index-url \S+|[A-Za-z0-9][\w.-]*(\[[\w,]+\])?((==|>=|<=|~=|!=|<|>)[\w.*]+(,|$|(?=\s)))*(\s+#.*)?)$/, l)
  }
})
