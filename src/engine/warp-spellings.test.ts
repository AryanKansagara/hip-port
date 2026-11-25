import test from 'node:test'
import assert from 'node:assert/strict'
import { mask } from './lex.ts'
import { detect } from './hazards.ts'
import { target } from './gfx.ts'

const rules = (src: string, t = 'gfx942') => detect(src, mask(src), target(t)).map(h => h.rule)

test('warp-size constants in any usual spelling', () => {
  for (const s of [
    '#define WARP_SIZE 32', '#define WARP_SIZE 32u', '#define WARPSIZE (32)', '#define kWarp 0x20',
    'const int WARP_SIZE = 32;', 'constexpr int kWarpSize = 32;', 'static constexpr uint32_t kWarpSize = 32u;', 'const unsigned int warp = 32U;',
    'static const std::size_t WARP = 32UL;', 'constexpr int WARP_SIZE{32};', 'const static unsigned int WARP_SIZE = (32);', 'enum { WARP_LANES = 32 };', 'enum Sizes : int { A = 1, WARP = 32, B };',
  ]) assert.deepEqual(rules(s), ['warp-size-32'], s)
})

test('lane arithmetic in any usual spelling', () => {
  for (const s of ['int a = threadIdx.x & 0x1f;', 'int a = threadIdx.x&0x1Fu;', 'int b = tid % 32u;', 'int c = lane / 32UL;', 'int d = threadIdx.y >> 5u;', 'if (tid%32==0) x();', 'f(tid/32)']) assert.deepEqual(rules(s), ['warp-size-32'], s)
})

test('reductions that start at 16 in any usual spelling', () => {
  const body = ' v += __shfl_down_sync(m, v, o, 32);'
  for (const s of ['for (int o = 16; o > 0; o >>= 1)', 'for (unsigned int o = 16; o > 0; o >>= 1)', 'for (uint32_t o = 16u; o; o /= 2)', 'for (std::size_t o = 0x10; o > 0; o >>= 1)', 'for (int o = 16; o > 0; o = o >> 1)', 'for (int o = 16; o > 0; o = o / 2)', 'for (o = 16; o > 0; o >>= 1u)', 'for (int o{16}; o > 0; o >>= 1)'])
    assert.deepEqual(rules(s + body), ['shfl-reduction-16'], s)
  assert.deepEqual(rules('for (unsigned int o = 16; o > 0; o >>= 1) v += a[o];'), [])
  assert.deepEqual(rules('for (int o = 160; o > 0; o >>= 1)' + body), [])
  assert.deepEqual(rules('for (int o = warpSize / 2; o > 0; o >>= 1)' + body), [])
})

test('numbers that only start like 32 are not flagged', () => {
  for (const s of ['#define WARP_SIZE 320', '#define WARP_F 32.5f', '#define WARP_SIZE\nint x = 32;', 'const int warp = 32 * n;'.replace('32 * n', 'n'), 'int x = idx / 320;', 'int x = idx & 0x1f0;', 'int x = idx & 0x1fu8;', 'int x = idx + 32;', 'int x = idx >> 50;', 'int x = threadIdx.x % warpSize;', 'enum { SIZE = 32 };', 'const int n = 32;', 'const float warp = 32.5f;', 'enum A { WARP_A = 1 }; int f() { int WARP = 32; }'])
    assert.deepEqual(rules(s), [], s)
})

test('severity still follows the target', () => {
  const src = 'const int WARP_SIZE = 32;'
  assert.equal(detect(src, mask(src), target('gfx942'))[0].severity, 'critical')
  assert.equal(detect(src, mask(src), target('gfx1201'))[0].severity, 'low')
})
