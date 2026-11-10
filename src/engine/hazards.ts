// Detectors for code that compiles under HIP but behaves differently on AMD GPUs, or does not
// compile at all. They run on masked code, so comments and strings never trigger them.
import { lineOf, matchBracket, splitArgs, type Masked } from './lex.ts'
import type { Target } from './gfx.ts'

export type Severity = 'critical' | 'high' | 'medium' | 'low'
export const SEVERITIES: Severity[] = ['critical', 'high', 'medium', 'low']

export interface Hazard {
  rule: string
  severity: Severity
  line: number
  /** The offending source line, trimmed */
  snippet: string
  message: string
  fix: string
}

interface Rule {
  id: string
  title: string
  /** Severity on wave64 targets (CDNA) and on wave32 targets (RDNA) */
  sev: [Severity, Severity]
  find(m: Masked, src: string): { at: number; message?: string }[]
  message: string
  fix: string
}

const all = (re: RegExp, s: string) => [...s.matchAll(re)].map(x => ({ at: x.index! }))

export const RULES: Rule[] = [
  {
    id: 'warp-size-32',
    title: 'Hard-coded warp size 32',
    sev: ['critical', 'low'],
    find: m => {
      const out: { at: number }[] = []
      for (const x of m.code.matchAll(/#\s*define\s+(\w*(?:WARP|warp|Warp)\w*)\s+32\b/g)) out.push({ at: x.index! })
      for (const x of m.code.matchAll(/\b(?:const|constexpr)\s+(?:static\s+)?(?:unsigned\s+|signed\s+)?\w+\s+\w*[Ww]arp\w*\s*=\s*32\b/g)) out.push({ at: x.index! })
      // Lane and warp index arithmetic: tid % 32, tid / 32, tid & 31, tid >> 5
      for (const x of m.code.matchAll(/\b(?:threadIdx\.[xyz]|\w*(?:tid|lane|Lane|idx|Idx|thread)\w*)\s*(?:%\s*32|\/\s*32|&\s*31|>>\s*5)\b/g)) out.push({ at: x.index! })
      return out
    },
    message: 'Assumes 32 threads per warp. AMD Instinct GPUs run 64-wide wavefronts, so lane and warp indices computed this way are wrong.',
    fix: 'Use the built-in warpSize (or a template parameter set from it) instead of 32, and & (warpSize - 1) instead of & 31.',
  },
  {
    id: 'shfl-reduction-16',
    title: 'Shuffle reduction starts at offset 16',
    sev: ['critical', 'low'],
    find: (m) => {
      const out: { at: number }[] = []
      for (const x of m.code.matchAll(/for\s*\(\s*(?:int|unsigned|auto)?\s*\w+\s*=\s*16\s*;[^;]*;[^)]*(?:>>=\s*1|\/=\s*2)\s*\)/g)) {
        const window = m.code.slice(x.index!, x.index! + 300)
        if (/__shfl/.test(window)) out.push({ at: x.index! })
      }
      return out
    },
    message: 'The butterfly starts at 16, so on a 64-lane wavefront lanes 32-63 are never folded into lane 0 and the warp sum is half the data.',
    fix: 'Start at warpSize / 2: for (int o = warpSize / 2; o > 0; o >>= 1).',
  },
  {
    id: 'mask-32bit',
    title: '32-bit lane mask',
    sev: ['high', 'low'],
    find: m => all(/__(?:shfl(?:_up|_down|_xor)?|ballot|any|all|syncwarp|match_any|match_all)_sync\s*\(\s*(?:0x[fF]{8}[uU]?L?|~0u?|FULL_MASK|0xffffffffu?)\b/g, m.code),
    message: 'HIP *_sync intrinsics take a 64-bit mask. 0xffffffff only names lanes 0-31 of a 64-lane wavefront.',
    fix: 'Pass __activemask() or a 64-bit constant such as 0xffffffffffffffffull (on wave32 targets the upper half is ignored).',
  },
  {
    id: 'shfl-no-width',
    title: 'Shuffle without width',
    sev: ['medium', 'low'],
    find: m => {
      const out: { at: number }[] = []
      for (const x of m.code.matchAll(/\b__shfl(_up|_down|_xor)?(_sync)?\s*\(/g)) {
        const open = x.index! + x[0].length - 1
        const close = matchBracket(m.code, open)
        if (close < 0) continue
        const n = splitArgs(m.code.slice(open + 1, close)).length
        if (n === (x[2] ? 3 : 2)) out.push({ at: x.index! })
      }
      return out
    },
    message: 'Width defaults to warpSize, which is 64 on CDNA. Code that treats each 32-lane half as its own group will exchange data across halves.',
    fix: 'Pass the width explicitly when the algorithm assumes 32-lane segments, or make the loop bounds depend on warpSize.',
  },
  {
    id: 'ballot-32bit',
    title: 'Ballot result truncated to 32 bits',
    sev: ['high', 'low'],
    find: m => [
      ...all(/\b(?:unsigned(?:\s+int)?|int|uint32_t|unsigned\s+long(?!\s+long))\s+\w+\s*=\s*__(?:ballot(?:_sync)?|activemask|match_any_sync)\s*\(/g, m.code),
      ...all(/\b__(?:popc|ffs|clz|brev)\s*\(\s*__(?:ballot(?:_sync)?|activemask)\b/g, m.code),
    ],
    message: '__ballot and __activemask return a 64-bit mask on AMD. Storing it in 32 bits or counting with __popc drops lanes 32-63.',
    fix: 'Store the mask as unsigned long long (or uint64_t) and use __popcll, __ffsll and __clzll.',
  },
  {
    id: 'lane-bit-32',
    title: '32-bit per-lane bit shift',
    sev: ['high', 'low'],
    find: m => all(/\b1[uU]?\s*<<\s*\(?\s*\w*(?:lane|Lane)\w*/g, m.code),
    message: '1u << lane is undefined for lanes 32-63.',
    fix: 'Use 1ull << lane and a 64-bit mask type.',
  },
  {
    id: 'warp-synchronous',
    title: 'Warp-synchronous code',
    sev: ['high', 'medium'],
    find: m => [
      ...all(/\bif\s*\(\s*\w+\s*<\s*32\s*\)\s*\{?[^}]{0,200}\bvolatile\b/g, m.code),
      ...all(/\bvolatile\s+\w+\s*\*\s*\w+\s*=\s*\w+\s*;[^}]{0,400}\+=\s*\w+\s*\[\s*\w+\s*\+\s*32\s*\]/g, m.code),
      ...all(/\b__syncwarp\s*\(/g, m.code),
    ],
    message: 'Relies on threads of a warp running in lockstep without a barrier, with a 32-wide tail. It is a data race the compiler may reorder, and the cut at 32 assumes 32-lane warps.',
    fix: 'Replace the unrolled tail with a shuffle reduction over warpSize, or keep __syncthreads() between steps.',
  },
  {
    id: 'inline-ptx',
    title: 'Inline PTX assembly',
    sev: ['critical', 'critical'],
    find: m => all(/\b(?:asm|__asm__)\s*(?:volatile\s*|__volatile__\s*)?\(/g, m.code),
    message: 'PTX does not exist on AMD GPUs. This block will not compile under hipcc.',
    fix: 'Use the HIP intrinsic or builtin that does the same thing (for example __lane_id(), __builtin_amdgcn_* or plain C++), guarded by #ifdef __HIP_PLATFORM_AMD__.',
  },
  {
    id: 'texture-reference',
    title: 'Texture reference',
    sev: ['high', 'high'],
    find: m => all(/\b(?:texture|surface)\s*<[^;>]*>\s*\w+\s*;/g, m.code),
    message: 'Texture and surface references are deprecated in HIP and unsupported in recent ROCm releases.',
    fix: 'Create a hipTextureObject_t with hipCreateTextureObject and pass it to the kernel; read with tex2D<float>(obj, x, y).',
  },
  {
    id: 'cooperative-groups',
    title: 'Cooperative groups',
    sev: ['medium', 'medium'],
    find: m => [
      ...all(/\bthis_grid\s*\(\s*\)/g, m.code),
      ...all(/\b(?:cg|cooperative_groups)\s*::\s*(?:reduce|memcpy_async|wait)\b/g, m.code),
      ...all(/\btiled_partition\s*<\s*(?:64|128)\s*>/g, m.code),
    ],
    message: 'Grid-wide sync needs a cooperative launch (hipLaunchCooperativeKernel) and every block resident at once; cg::reduce and cg::memcpy_async are not in HIP; tiles wider than the wavefront fail on wave32 targets.',
    fix: 'Launch with hipLaunchCooperativeKernel and size the grid with hipOccupancyMaxActiveBlocksPerMultiprocessor; replace cg::reduce with a shuffle loop or hipcub::WarpReduce.',
  },
  {
    id: 'unsupported-intrinsic',
    title: 'Intrinsic without a HIP equivalent',
    sev: ['high', 'high'],
    find: m => [...m.code.matchAll(/\b(__reduce_(?:add|min|max|and|or|xor)_sync|__cvta_generic_to_shared|__pipeline_memcpy_async|__pipeline_commit|__pipeline_wait_prior|__vimax3_\w+|__viaddmax_\w+|__cluster_dims__|__ldcs|__ldlu|__ldcv|__stcs|__stwt)\b/g)].map(x => ({ at: x.index!, message: `${x[1]} has no HIP equivalent as of ROCm 7.0.` })),
    message: 'This intrinsic has no HIP equivalent.',
    fix: 'Rewrite with portable code: a shuffle loop for warp reductions, plain loads plus __syncthreads() for async copies, or __builtin_nontemporal_load for streaming hints.',
  },
  {
    id: 'wmma',
    title: 'WMMA tensor-core fragments',
    sev: ['high', 'high'],
    find: m => all(/\b(?:nvcuda\s*::\s*)?wmma\s*::\s*(?:fragment|load_matrix_sync|mma_sync)\b/g, m.code),
    message: 'rocWMMA mirrors the API but supported shapes and types differ by target (MFMA on CDNA, WMMA on RDNA3+); gfx1030 has no matrix cores.',
    fix: 'Port to rocwmma:: with shapes from the rocWMMA support table, or call hipBLASLt for the GEMM.',
  },
  {
    id: 'cuda-arch-macro',
    title: '__CUDA_ARCH__ guard',
    sev: ['medium', 'medium'],
    find: m => all(/\b__CUDA_ARCH__\b/g, m.code),
    message: 'hipcc does not define __CUDA_ARCH__, so device-only branches guarded by it silently disappear.',
    fix: 'Use __HIP_DEVICE_COMPILE__ for device-pass checks and __gfx942__ / __GFX9__ style macros for architecture checks.',
  },
  {
    id: 'compute-capability',
    title: 'Compute capability branch',
    sev: ['medium', 'medium'],
    find: m => all(/\.\s*major\s*(?:>=|>|==|<=|<)\s*\d+/g, m.code),
    message: 'AMD GPUs report a pseudo compute capability; feature checks on it pick the wrong path.',
    fix: 'Branch on hipDeviceProp_t::gcnArchName (for example "gfx942") or query the specific attribute.',
  },
  {
    id: 'dynamic-parallelism',
    title: 'Kernel launched from device code',
    sev: ['critical', 'critical'],
    find: m => {
      const out: { at: number }[] = []
      for (const x of m.code.matchAll(/\b__(?:global|device)__\b[^;{]*\{/g)) {
        const open = x.index! + x[0].length - 1
        const close = matchBracket(m.code, open)
        if (close < 0) continue
        const body = m.code.slice(open, close)
        const k = body.indexOf('<<<')
        if (k >= 0) out.push({ at: open + k })
      }
      return out
    },
    message: 'Dynamic parallelism (launching kernels from the GPU) is not supported by HIP.',
    fix: 'Move the child launch to the host, or flatten the work into the parent kernel with a work queue.',
  },
]

export const RULE_BY_ID = Object.fromEntries(RULES.map(r => [r.id, r]))

export function detect(src: string, m: Masked, t: Target): Hazard[] {
  const lines = src.split('\n')
  const out: Hazard[] = []
  const seen = new Set<string>()
  for (const r of RULES) {
    for (const hit of r.find(m, src)) {
      const line = lineOf(m, hit.at)
      const key = r.id + ':' + line
      if (seen.has(key)) continue
      seen.add(key)
      out.push({ rule: r.id, severity: r.sev[t.wave === 64 ? 0 : 1], line, snippet: lines[line - 1].trim().slice(0, 160), message: hit.message ?? r.message, fix: r.fix })
    }
  }
  return out.sort((a, b) => a.line - b.line || SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity))
}
