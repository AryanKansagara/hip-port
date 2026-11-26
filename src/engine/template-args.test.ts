import test from 'node:test'
import assert from 'node:assert/strict'
import { splitArgs, mask } from './lex.ts'
import { translate } from './translate.ts'
import { detect } from './hazards.ts'
import { target } from './gfx.ts'

const split = (s: string) => splitArgs(s).map(a => a.text.trim())
const t = (s: string, ggl = false) => translate('#include <hip/hip_runtime.h>\n' + s, { ggl }).out.split('\n')[1]

test('template argument lists are one argument', () => {
  assert.deepEqual(split('kern<float, 4>, attr, n'), ['kern<float, 4>', 'attr', 'n'])
  assert.deepEqual(split('static_cast<std::pair<int, int>*>(p), tbl'), ['static_cast<std::pair<int, int>*>(p)', 'tbl'])
  assert.deepEqual(split('f<A<B, C>, D>(x, y), z'), ['f<A<B, C>, D>(x, y)', 'z'])
  assert.deepEqual(split('f<(n > 3), 2>::value, z'), ['f<(n > 3), 2>::value', 'z'])
  assert.deepEqual(split('cast<T<a, b>&>(v), cast<T<a, b> const*>(v)'), ['cast<T<a, b>&>(v)', 'cast<T<a, b> const*>(v)'])
})

test('comparisons and shifts still split at every comma', () => {
  for (const s of ['a < b, c > d', 'a<b, c>d', 'a<b, c>=d', 'n << 3, m >> 2', 'n<<3, m>>2', 'a < b ? x : y, c > d', 'a<b && c, d>(e)', 'p->x, q->y', 'a[i]<b, c>(d)', '3<b, c>(d)', 'a<b, c', 'a<b, c>>1', 'p->x<y, z>(w)'])
    assert.deepEqual(split(s), s.split(',').map(x => x.trim()), s)
})

test('host call fix-ups and launches keep template arguments whole', () => {
  assert.equal(t('cudaFuncSetAttribute(kern<float, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, n);'), 'hipFuncSetAttribute(reinterpret_cast<const void*>(kern<float, 4>), hipFuncAttributeMaxDynamicSharedMemorySize, n);')
  assert.equal(t('cudaMemcpyFromSymbol(static_cast<std::pair<int,int>*>(p), tbl, 4);'), 'hipMemcpyFromSymbol(static_cast<std::pair<int,int>*>(p), HIP_SYMBOL(tbl), 4);')
  assert.equal(t('k<<<div_up<int, 8>(n), 256, (a < b)>>>(x);', true), 'hipLaunchKernelGGL(k, dim3(div_up<int, 8>(n)), dim3(256), (a < b), 0, x);')
  assert.equal(t('k<<<n >> 3, m << 1>>>(x);', true), 'hipLaunchKernelGGL(k, dim3(n >> 3), dim3(m << 1), 0, 0, x);')
})

test('a dropped callback argument goes even when it holds a CUDA name', () => {
  assert.equal(t('cuOccupancyMaxPotentialBlockSize(&g, &b, f, (CUoccupancyB2DSize)0, smem, limit);'), 'hipModuleOccupancyMaxPotentialBlockSize(&g, &b, f, smem, limit);')
})

test('a template call inside a shuffle is not counted as a width', () => {
  const src = 'v += __shfl_down_sync(m, get<int, 4>(v), 1);'
  assert.ok(detect(src, mask(src), target('gfx942')).some(h => h.rule === 'shfl-no-width'))
})
