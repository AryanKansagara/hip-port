// AMD GPU targets (LLVM gfx names) with the facts that change porting decisions.
// Checked against the ROCm 7.0 compatibility matrix and LLVM AMDGPU docs, November 2026.

export interface Target {
  id: string
  family: 'CDNA' | 'RDNA'
  gpus: string
  /** Default wavefront size under hipcc */
  wave: 32 | 64
  /** Matrix cores available to HIP code */
  matrix: string
  fp8: 'none' | 'fnuz' | 'ocp'
  rocm: 'supported' | 'consumer' | 'community'
  lds: number
  note: string
}

export const TARGETS: Target[] = [
  { id: 'gfx942', family: 'CDNA', gpus: 'Instinct MI300X, MI300A, MI325X', wave: 64, matrix: 'MFMA', fp8: 'fnuz', rocm: 'supported', lds: 64, note: 'Wave64 only. FP8 uses the FNUZ encodings, so CUDA e4m3 checkpoints need conversion. XNACK is off by default; managed memory needs HSA_XNACK=1.' },
  { id: 'gfx950', family: 'CDNA', gpus: 'Instinct MI350X, MI355X', wave: 64, matrix: 'MFMA', fp8: 'ocp', rocm: 'supported', lds: 160, note: 'Wave64 only. OCP FP8 and MXFP4/MXFP6. 160 KB LDS per CU.' },
  { id: 'gfx90a', family: 'CDNA', gpus: 'Instinct MI210, MI250, MI250X', wave: 64, matrix: 'MFMA', fp8: 'none', rocm: 'supported', lds: 64, note: 'Wave64 only. No FP8 hardware: fp8 kernels need a fallback path.' },
  { id: 'gfx1201', family: 'RDNA', gpus: 'Radeon RX 9070 XT, Radeon AI PRO R9700', wave: 32, matrix: 'WMMA (RDNA4)', fp8: 'ocp', rocm: 'consumer', lds: 64, note: 'Wave32 by default (wave64 with -mwavefrontsize64). WMMA with OCP FP8. Many int4/fp8 kernels in serving stacks are still CDNA-only.' },
  { id: 'gfx1100', family: 'RDNA', gpus: 'Radeon RX 7900 XTX / XT, Radeon PRO W7900', wave: 32, matrix: 'WMMA (RDNA3)', fp8: 'none', rocm: 'consumer', lds: 64, note: 'Wave32 by default. No FP8. Officially supported for PyTorch on Linux and on Windows via WSL.' },
  { id: 'gfx1030', family: 'RDNA', gpus: 'Radeon RX 6800 / 6900, PRO W6800', wave: 32, matrix: 'none', fp8: 'none', rocm: 'community', lds: 64, note: 'Wave32 by default. No matrix cores. Works with HSA_OVERRIDE_GFX_VERSION tricks but is not on the current support list.' },
]

export const target = (id: string) => TARGETS.find(t => t.id === id) ?? TARGETS[0]
