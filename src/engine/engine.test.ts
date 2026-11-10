import test from 'node:test'
import assert from 'node:assert/strict'
import { mask, lineOf } from './lex.ts'
import { lookup, lookupHeader } from './mapping.ts'
import { translate, findLaunches } from './translate.ts'
import { detect } from './hazards.ts'
import { target } from './gfx.ts'
import { parseRequirements, parseCondaEnv, rocmRequirements } from './pydeps.ts'
import { readZip, writeZip, crc32 } from './zip.ts'
import { scan, diffLines, worklistCsv, readiness, snapshotOf, compareSnapshots, coveragePct } from './scan.ts'
import { SAMPLES, HERO_KERNEL } from './sample.ts'

const rules = (src: string, t = 'gfx942') => detect(src, mask(src), target(t)).map(h => h.rule)

test('mask blanks comments and strings but keeps offsets and newlines', () => {
  const src = 'a /* cudaMalloc */ b // cudaFree\n"cudaMemcpy" c'
  const m = mask(src)
  assert.equal(m.code.length, src.length)
  assert.ok(!m.code.includes('cudaMalloc') && !m.code.includes('cudaFree') && !m.code.includes('cudaMemcpy'))
  assert.equal(lineOf(m, src.indexOf('c', 40)), 2)
})

test('mask keeps include header names visible, and digit separators are not char literals', () => {
  const m = mask('#include "cublas_v2.h"\nint x = 1\'000\'000; cudaFree(p);')
  assert.ok(m.code.includes('cublas_v2.h'))
  assert.ok(m.code.includes('cudaFree'))
})

test('translator skips API names inside comments and string literals', () => {
  const { out } = translate('// call cudaMalloc here\nprintf("cudaMalloc failed");\ncudaMalloc(&p, n);')
  assert.match(out, /\/\/ call cudaMalloc here/)
  assert.match(out, /"cudaMalloc failed"/)
  assert.match(out, /hipMalloc\(&p, n\)/)
})

test('runtime, enum and type names translate through the family rule', () => {
  const { out } = translate('cudaStream_t s; cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s); cudaError_t e = cudaGetLastError();')
  assert.match(out, /hipStream_t s; hipMemcpyAsync\(d, h, n, hipMemcpyHostToDevice, s\); hipError_t e = hipGetLastError\(\)/)
})

test('explicit table entries override the family rule', () => {
  assert.equal(lookup('cudaThreadSynchronize')?.hip, 'hipDeviceSynchronize')
  assert.equal(lookup('cudaErrorMemoryAllocation')?.hip, 'hipErrorOutOfMemory')
  assert.equal(lookup('cuMemAlloc')?.hip, 'hipMalloc')
  assert.equal(lookup('cuLaunchKernel')?.hip, 'hipModuleLaunchKernel')
})

test('library families: cuBLAS (with _v2), cuFFT, cuRAND device API, cuSPARSE, NCCL', () => {
  assert.equal(lookup('cublasSgemm_v2')?.hip, 'hipblasSgemm')
  assert.equal(lookup('CUBLAS_OP_T')?.hip, 'HIPBLAS_OP_T')
  assert.equal(lookup('cufftExecC2C')?.hip, 'hipfftExecC2C')
  assert.equal(lookup('curand_uniform')?.hip, 'hiprand_uniform')
  assert.equal(lookup('curandState')?.hip, 'hiprandState')
  assert.equal(lookup('cusparseSpMV')?.hip, 'hipsparseSpMV')
  assert.equal(lookup('ncclAllReduce')?.hip, 'ncclAllReduce')
  assert.equal(lookup('ncclAllReduce')?.lib, 'nccl')
})

test('cuDNN maps plumbing to MIOpen and flags the rest as needing a hand port', () => {
  assert.equal(lookup('cudnnCreate')?.hip, 'miopenCreate')
  assert.equal(lookup('cudnnConvolutionForward')?.status, 'partial')
  const fused = lookup('cudnnConvolutionBiasActivationForward')
  assert.equal(fused?.hip, null)
  assert.equal(fused?.status, 'unsupported')
})

test('headers translate, including quoted includes, cub and thrust', () => {
  const { out, uses } = translate('#include <cuda_runtime.h>\n#include "cublas_v2.h"\n#include <cub/cub.cuh>\n#include <thrust/sort.h>\n')
  assert.match(out, /#include <hip\/hip_runtime.h>/)
  assert.match(out, /#include "hipblas\/hipblas.h"/)
  assert.match(out, /#include <hipcub\/hipcub.hpp>/)
  assert.match(out, /#include <thrust\/sort.h>/)
  assert.equal(lookupHeader('cudnn.h')?.hip, 'miopen/miopen.h')
  assert.ok(uses.some(u => u.kind === 'header' && u.lib === 'thrust'))
})

test('cub:: namespace becomes hipcub:: and adds the runtime include when missing', () => {
  const { out, addedInclude } = translate('// header comment\n__global__ void k() { using R = cub::BlockReduce<float, 128>; }')
  assert.match(out, /hipcub::BlockReduce/)
  assert.ok(addedInclude)
  assert.match(out, /^\/\/ header comment\n#include <hip\/hip_runtime.h>\n/)
})

test('project-defined macros and helpers are never renamed', () => {
  const src = '#define cudaCheck(x) x\ncudaError_t cudaWrap(int a) { return cudaSuccess; }\ncudaCheck(cudaWrap(1));'
  const { out } = translate(src)
  assert.match(out, /#define cudaCheck\(x\) x/)
  assert.match(out, /hipError_t cudaWrap\(int a\) \{ return hipSuccess; \}/)
  assert.match(out, /cudaCheck\(cudaWrap\(1\)\)/)
})

test('unknown CUDA-looking names are reported as unmapped, not renamed', () => {
  const { out, uses } = translate('CU_CTX_SCHED_BLOCKING_SYNC; CUDA_ERROR_LAUNCH_TIMEOUT;')
  assert.match(out, /CU_CTX_SCHED_BLOCKING_SYNC/)
  assert.deepEqual(uses.filter(u => u.status === 'unmapped').map(u => u.cuda).sort(), ['CUDA_ERROR_LAUNCH_TIMEOUT', 'CU_CTX_SCHED_BLOCKING_SYNC'])
})

test('member access is not translated', () => {
  const { out } = translate('cfg.cudaGraph = 1; p->cudaStream = 0;')
  assert.equal(out, 'cfg.cudaGraph = 1; p->cudaStream = 0;')
})

test('launch conversion handles templates, defaults and nested parentheses', () => {
  const src = 'k<<<grid, block>>>(a, f(b, c));\nns::tk<float, 4><<<dim3(g, 1), 256, smem, stream>>>(x);'
  assert.equal(findLaunches(mask(src)).length, 2)
  const { out } = translate(src, { ggl: true })
  assert.match(out, /hipLaunchKernelGGL\(k, dim3\(grid\), dim3\(block\), 0, 0, a, f\(b, c\)\);/)
  assert.match(out, /hipLaunchKernelGGL\(\(ns::tk<float, 4>\), dim3\(dim3\(g, 1\)\), dim3\(256\), smem, stream, x\);/)
})

test('launch conversion applies identifier renames inside the arguments', () => {
  const { out } = translate('k<<<1, 1, 0, s>>>(cudaStreamPerThread);', { ggl: true })
  assert.match(out, /hipLaunchKernelGGL\(k, dim3\(1\), dim3\(1\), 0, s, hipStreamPerThread\)/)
  // Without ggl the chevrons stay; hipcc accepts them.
  assert.match(translate('k<<<1, 1>>>(x);').out, /k<<<1, 1>>>\(x\)/)
})

test('warp-size hazards: hard-coded 32, lane math, reduction from 16', () => {
  const r = rules('#define WARP_SIZE 32\nint lane = threadIdx.x % 32;\nfor (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);')
  assert.ok(r.includes('warp-size-32'))
  assert.ok(r.includes('shfl-reduction-16'))
  assert.ok(r.includes('mask-32bit'))
  assert.ok(r.includes('shfl-no-width'))
})

test('hazard severity depends on the target wavefront size', () => {
  const src = 'int lane = threadIdx.x & 31;'
  assert.equal(detect(src, mask(src), target('gfx942'))[0].severity, 'critical')
  assert.equal(detect(src, mask(src), target('gfx1100'))[0].severity, 'low')
})

test('shuffle with an explicit width is not flagged for width', () => {
  assert.ok(!rules('v += __shfl_down_sync(m, v, 1, 16);').includes('shfl-no-width'))
  assert.ok(rules('v += __shfl_down(v, 1);').includes('shfl-no-width'))
})

test('ballot truncation, popc on a ballot and 32-bit lane shifts are flagged', () => {
  assert.ok(rules('unsigned m = __ballot_sync(full, p);').includes('ballot-32bit'))
  assert.ok(rules('int c = __popc(__ballot_sync(full, p));').includes('ballot-32bit'))
  assert.ok(!rules('unsigned long long m = __ballot_sync(full, p);').includes('ballot-32bit'))
  assert.ok(rules('return (1u << laneId()) - 1;').includes('lane-bit-32'))
})

test('PTX, texture references, device-side launches and sm_80 intrinsics are flagged', () => {
  const r = rules('asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));\ntexture<float, 2> t;\n__global__ void p() { c<<<1, 1>>>(); }\nx = __reduce_add_sync(m, x);')
  for (const id of ['inline-ptx', 'texture-reference', 'dynamic-parallelism', 'unsupported-intrinsic']) assert.ok(r.includes(id), id)
})

test('hazards inside comments and strings do not fire', () => {
  assert.deepEqual(rules('// asm("trap;"); lane % 32\nconst char* s = "texture<float, 2> t;";'), [])
})

test('warp-synchronous volatile tail and __CUDA_ARCH__ guards are flagged', () => {
  const r = rules('if (tid < 32) { volatile float* vs = s; vs[tid] += vs[tid + 32]; }\n#if __CUDA_ARCH__ >= 800\n#endif')
  assert.ok(r.includes('warp-synchronous'))
  assert.ok(r.includes('cuda-arch-macro'))
})

test('requirements parser: CUDA index, +cu pins, extras, comments, git urls', () => {
  const r = parseRequirements('--extra-index-url https://download.pytorch.org/whl/cu121\ntorch==2.3.1+cu121  # pinned\nflash_attn==2.5\nfoo[bar]>=1 ; python_version>"3.8"\ngit+https://github.com/x/y.git#egg=mylib\nnvidia-cublas-cu12\n-r dev.txt')
  assert.equal(r.cudaIndexes.length, 1)
  const torch = r.deps.find(d => d.name === 'torch')!
  assert.equal(torch.cudaLocal, 'cu121')
  assert.equal(r.deps.find(d => d.name === 'flash_attn')?.status, 'fork')
  assert.equal(r.deps.find(d => d.name === 'foo')?.spec, '[bar]>=1')
  assert.equal(r.deps.find(d => d.name === 'mylib')?.status, 'unknown')
  assert.equal(r.deps.find(d => d.name === 'nvidia-cublas-cu12')?.status, 'cuda-only')
  assert.equal(r.errors[0].line, 7)
})

test('conda env parser reads channels, conda deps and the nested pip list', () => {
  const r = parseCondaEnv('name: x\nchannels:\n  - nvidia\n  - conda-forge\ndependencies:\n  - pytorch=2.4.0\n  - pytorch-cuda=12.1\n  - pip:\n      - vllm==0.5.3\n  - numpy\n')
  assert.deepEqual(r.cudaChannels, ['nvidia'])
  assert.deepEqual(r.deps.map(d => [d.name, d.source]), [['pytorch', 'conda'], ['pytorch-cuda', 'conda'], ['vllm', 'pip'], ['numpy', 'conda']])
  assert.equal(r.deps.find(d => d.name === 'vllm')?.line, 9)
})

test('ROCm requirements rewrite swaps the index and removes CUDA-only packages', () => {
  const out = rocmRequirements(parseRequirements('--extra-index-url https://download.pytorch.org/whl/cu121\ntorch==2.3.1+cu121\nonnxruntime-gpu\nnvidia-ml-py\ntransformers>=4'))
  assert.match(out, /--extra-index-url https:\/\/download.pytorch.org\/whl\/rocm/)
  assert.match(out, /^torch==2\.3\.1$/m)
  assert.match(out, /^onnxruntime-rocm$/m)
  assert.ok(!/cu121/.test(out))
  assert.match(out, /^transformers>=4$/m)
})

test('zip round trip: write stored entries and read them back with CRC', async () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926)
  const z = writeZip([{ path: 'a/b.hip', text: 'hello ✓' }, { path: 'c.txt', text: '' }])
  const back = await readZip(z)
  assert.deepEqual(back.map(e => [e.path, new TextDecoder().decode(e.data)]), [['a/b.hip', 'hello ✓'], ['c.txt', '']])
  await assert.rejects(readZip(new Uint8Array(40)), /Not a zip/)
})

test('zip reader inflates deflate-compressed entries', async () => {
  // Build a deflated entry by hand using the platform CompressionStream.
  const data = new TextEncoder().encode('__global__ void k() {}\n'.repeat(50))
  const comp = new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new CompressionStream('deflate-raw'))).arrayBuffer())
  const stored = writeZip([{ path: 'k.cu', text: '' }])
  // Patch the stored archive into a deflated one: rebuild with method 8.
  const name = new TextEncoder().encode('k.cu')
  const crc = crc32(data)
  const lh = new DataView(new ArrayBuffer(30)); lh.setUint32(0, 0x04034b50, true); lh.setUint16(8, 8, true); lh.setUint32(14, crc, true); lh.setUint32(18, comp.length, true); lh.setUint32(22, data.length, true); lh.setUint16(26, name.length, true)
  const ch = new DataView(new ArrayBuffer(46)); ch.setUint32(0, 0x02014b50, true); ch.setUint16(10, 8, true); ch.setUint32(16, crc, true); ch.setUint32(20, comp.length, true); ch.setUint32(24, data.length, true); ch.setUint16(28, name.length, true)
  const cdOff = 30 + name.length + comp.length
  const eo = new DataView(new ArrayBuffer(22)); eo.setUint32(0, 0x06054b50, true); eo.setUint16(8, 1, true); eo.setUint16(10, 1, true); eo.setUint32(12, 46 + name.length, true); eo.setUint32(16, cdOff, true)
  const parts = [new Uint8Array(lh.buffer), name, comp, new Uint8Array(ch.buffer), name, new Uint8Array(eo.buffer)]
  const buf = new Uint8Array(parts.reduce((s, p) => s + p.length, 0)); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length }
  const [e] = await readZip(buf)
  assert.equal(e.path, 'k.cu')
  assert.equal(new TextDecoder().decode(e.data), new TextDecoder().decode(data))
  assert.ok(stored.length > 0)
})

test('sample mini-repo scan finds every planted hazard class on gfx942', () => {
  const s = SAMPLES[0]
  const r = scan(s.files.filter(f => !f.path.endsWith('.txt')), s.files.find(f => f.path.endsWith('.txt'))!, { target: 'gfx942', ggl: false })
  const found = new Set(r.files.flatMap(f => f.hazards.map(h => h.rule)))
  for (const id of ['warp-size-32', 'shfl-reduction-16', 'mask-32bit', 'ballot-32bit', 'lane-bit-32', 'warp-synchronous', 'inline-ptx', 'texture-reference', 'cooperative-groups', 'cuda-arch-macro']) assert.ok(found.has(id), id)
  assert.ok(r.libs.some(l => l.lib === 'cublas') && r.libs.some(l => l.lib === 'cub') && r.libs.some(l => l.lib === 'thrust'))
  assert.equal(r.files.find(f => f.path === 'src/reduce.cu')?.outPath, 'src/reduce.hip')
  // The project's CUDA_CHECK macro survives.
  assert.match(r.files.find(f => f.path === 'src/gemm.cu')!.translated, /hipblasSgemm\(handle, HIPBLAS_OP_N/)
  assert.ok(r.counts.critical > 0 && r.items.some(i => i.category === 'dependency' && i.severity === 'high'))
  assert.equal(r.checklist.find(c => c.id === 'torch')?.state, 'todo')
  assert.ok(coveragePct(r.coverage) > 0.8)
})

test('the hero kernel translates and lights up the expected lines', () => {
  const t = translate(HERO_KERNEL)
  assert.match(t.out, /hipMemsetAsync/)
  assert.match(t.out, /hipStreamSynchronize/)
  const hz = detect(HERO_KERNEL, mask(HERO_KERNEL), target('gfx942'))
  assert.ok(hz.length >= 5)
  assert.ok(hz.every(h => h.line >= 1 && h.line <= HERO_KERNEL.split('\n').length))
})

test('line diff pairs changed lines and keeps unchanged ones aligned', () => {
  const rows = diffLines('a\nb\nc\nd', 'a\nB\nc\nx\nd')
  assert.deepEqual(rows.map(r => r.type), ['same', 'mod', 'same', 'add', 'same'])
  assert.equal(rows[3].b, 4)
})

test('worklist CSV escapes quotes and commas; readiness rises as items are resolved', () => {
  const items = [{ key: 'k1', severity: 'critical' as const, category: 'hazard' as const, file: 'a.cu', line: 3, title: 'x, "y"', detail: 'd', fix: 'f' }, { key: 'k2', severity: 'low' as const, category: 'api' as const, file: 'a.cu', line: 9, title: 't', detail: '', fix: '' }]
  const csv = worklistCsv(items, k => k === 'k1' ? 'done' : 'open')
  assert.match(csv, /"x, ""y"""/)
  assert.match(csv, /,done\n/)
  assert.ok(readiness(items, k => k === 'k1') > readiness(items))
  assert.equal(readiness([]), 100)
})

test('snapshot compare reports fixed and new items by stable key', () => {
  const s = SAMPLES[0]
  const src = s.files.filter(f => f.path.endsWith('.cu') || f.path.endsWith('.cuh'))
  const before = scan(src, null, { target: 'gfx942', ggl: false })
  const fixed = src.map(f => f.path === 'src/common.cuh' ? { ...f, text: f.text.replace(/asm volatile\([^;]*;/, 'id = __lane_id();') } : f)
  const after = scan(fixed, null, { target: 'gfx942', ggl: false })
  const c = compareSnapshots(snapshotOf(before, 'a', 'one', 1), snapshotOf(after, 'b', 'two', 2))
  assert.ok(c.fixed.some(i => i.key.includes('inline-ptx')))
  assert.equal(c.added.length, 0)
  assert.ok(c.scoreDelta > 0)
})

test('llm sample on gfx1201: fp8 and warp intrinsic flagged, warp-32 lane math only low', () => {
  const s = SAMPLES[1]
  const r = scan(s.files.filter(f => f.path.endsWith('.cu')), s.files.find(f => f.path.endsWith('.yml'))!, { target: 'gfx1201', ggl: true })
  const hz = r.files.flatMap(f => f.hazards)
  assert.ok(hz.some(h => h.rule === 'unsupported-intrinsic' && h.severity === 'high'))
  assert.ok(hz.filter(h => h.rule === 'warp-size-32').every(h => h.severity === 'low'))
  assert.ok(r.files[0].translated.includes('__hip_fp8_e4m3'))
  assert.ok(r.files[1].translated.includes('hipLaunchKernelGGL(sampleRows'))
  assert.ok(r.files[1].translated.includes('roctxRangePushA'))
  assert.ok(r.deps!.cudaChannels.includes('nvidia'))
})
