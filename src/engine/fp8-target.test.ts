import test from 'node:test'
import assert from 'node:assert/strict'
import { scan } from './scan.ts'
import { TARGETS } from './gfx.ts'
import { translate } from './translate.ts'

const SRC = '#include <hip/hip_runtime.h>\n__nv_fp8_e4m3 a; __nv_fp8_e5m2 b; __nv_fp8x2_e4m3 c; __nv_fp8x4_e5m2 d; e = __NV_E4M3; __nv_fp8x2_storage_t f; __nv_fp8_e8m0 g;\n'
const run = (target: string) => scan([{ path: 'k.cu', text: SRC }], null, { target, ggl: false })
const line = (target: string) => run(target).files[0].translated.split('\n')[1]
const FNUZ = '__hip_fp8_e4m3_fnuz a; __hip_fp8_e5m2_fnuz b; __hip_fp8x2_e4m3_fnuz c; __hip_fp8x4_e5m2_fnuz d; e = __HIP_E4M3_FNUZ; __hip_fp8x2_storage_t f; __hip_fp8_e8m0 g;'
const OCP = '__hip_fp8_e4m3 a; __hip_fp8_e5m2 b; __hip_fp8x2_e4m3 c; __hip_fp8x4_e5m2 d; e = __HIP_E4M3; __hip_fp8x2_storage_t f; __hip_fp8_e8m0 g;'

test('FP8 names follow the target format', () => {
  for (const t of TARGETS) assert.equal(line(t.id), t.fp8 === 'ocp' ? OCP : FNUZ, t.id)
  assert.equal(translate(SRC).out.split('\n')[1], FNUZ)
})

test('the work items say what was emitted', () => {
  const fnuz = run('gfx942').items.filter(i => i.category === 'api')
  assert.deepEqual(fnuz.map(i => i.severity), ['low', 'low', 'low', 'low', 'low'])
  assert.match(fnuz[0].title, /→ __hip_fp8\w*_fnuz needs review/)
  assert.deepEqual(run('gfx950').items.filter(i => i.category === 'api'), [])
})
