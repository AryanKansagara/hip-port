// Python dependency readiness for ROCm: parses requirements.txt / environment.yml and checks each
// package against a curated table. Every row carries the date it was last checked.

export type DepStatus = 'ok' | 'rocm' | 'fork' | 'partial' | 'cuda-only' | 'unknown'

export interface Known { status: DepStatus; note: string; action: string; replace?: string; checked: string }

export const ROCM_INDEX = 'https://download.pytorch.org/whl/rocm7.0'

const K: Record<string, Known> = {
  torch: { status: 'rocm', note: 'Official ROCm wheels on the PyTorch index. torch.cuda.* keeps working; torch.version.hip is set.', action: `Install from ${ROCM_INDEX} and drop any +cuXXX local version.`, checked: '2026-10' },
  torchvision: { status: 'rocm', note: 'Built alongside torch on the ROCm index.', action: 'Install from the same ROCm index as torch.', checked: '2026-10' },
  torchaudio: { status: 'rocm', note: 'Built alongside torch on the ROCm index.', action: 'Install from the same ROCm index as torch.', checked: '2026-10' },
  triton: { status: 'ok', note: 'Upstream Triton has an AMD backend; ROCm torch wheels pull in pytorch-triton-rocm.', action: 'Remove the explicit pin and let torch choose pytorch-triton-rocm.', checked: '2026-10' },
  'pytorch-triton': { status: 'cuda-only', note: 'The CUDA build of Triton shipped with CUDA torch.', action: 'Remove; ROCm torch depends on pytorch-triton-rocm.', replace: '', checked: '2026-10' },
  'bitsandbytes': { status: 'partial', note: 'ROCm backend is upstream: LLM.int8 and 4-bit NF4/FP4 work on CDNA and RDNA3+, some paged optimizers do not.', action: 'Use bitsandbytes >= 0.48 and verify your optimizer; on gfx90a run the test suite first.', checked: '2026-10' },
  'flash-attn': { status: 'fork', note: 'AMD maintains ROCm/flash-attention: Composable Kernel backend for MI200/MI300/MI350, Triton backend for RDNA3/4.', action: 'Build from github.com/ROCm/flash-attention, or use torch SDPA which dispatches to AOTriton.', replace: '# flash-attn: build from https://github.com/ROCm/flash-attention (CK backend on Instinct)', checked: '2026-10' },
  'flash_attn': { status: 'fork', note: 'Same as flash-attn.', action: 'Build from github.com/ROCm/flash-attention.', replace: '# flash-attn: build from https://github.com/ROCm/flash-attention (CK backend on Instinct)', checked: '2026-10' },
  xformers: { status: 'rocm', note: 'ROCm builds are published on the PyTorch ROCm index; memory-efficient attention uses CK on Instinct, some ops fall back.', action: 'Install from the PyTorch ROCm index matching torch.', checked: '2026-10' },
  vllm: { status: 'fork', note: 'Supported on Instinct via rocm/vllm images and source builds. On RDNA4 (gfx1201) several int4/fp8 quantized kernels (Marlin, Machete) are not written yet.', action: 'Use the rocm/vllm Docker image for MI300/MI350; on Radeon prefer fp16/bf16 or GPTQ via the Triton path.', replace: '# vllm: use the rocm/vllm Docker image or build with VLLM_TARGET_DEVICE=rocm', checked: '2026-10' },
  sglang: { status: 'fork', note: 'ROCm images (lmsysorg/sglang:*-rocm) for MI300/MI350.', action: 'Use the ROCm Docker image.', checked: '2026-10' },
  flashinfer: { status: 'cuda-only', note: 'No ROCm build; kernels are CUDA-specific.', action: 'Use the attention backend your server provides on ROCm (AITER / Triton).', replace: '', checked: '2026-10' },
  deepspeed: { status: 'ok', note: 'Builds its ops with hipify at install time on ROCm torch.', action: 'Install after ROCm torch; set DS_BUILD_OPS=0 to JIT-build only what you use.', checked: '2026-10' },
  apex: { status: 'fork', note: 'AMD maintains ROCm/apex.', action: 'Build from github.com/ROCm/apex, or drop it: fused Adam and LayerNorm are in torch.', checked: '2026-10' },
  'auto-gptq': { status: 'partial', note: 'Unmaintained; the ROCm wheels stopped at older torch versions.', action: 'Move to gptqmodel, which has a ROCm path, or load GPTQ through transformers + optimum.', checked: '2026-10' },
  autoawq: { status: 'partial', note: 'Archived upstream; AWQ GEMM kernels are CUDA-only, the Triton path works slower.', action: 'Use vLLM on ROCm for AWQ serving, or quantize to GPTQ.', checked: '2026-10' },
  'llama-cpp-python': { status: 'ok', note: 'Builds with the HIP backend.', action: 'pip install with CMAKE_ARGS="-DGGML_HIP=on -DAMDGPU_TARGETS=<gfx>".', checked: '2026-10' },
  'mamba-ssm': { status: 'partial', note: 'Builds from source on ROCm torch; selective-scan kernels are hipified at build time.', action: 'Install with --no-build-isolation after ROCm torch; set PYTORCH_ROCM_ARCH.', checked: '2026-10' },
  'causal-conv1d': { status: 'partial', note: 'Builds from source on ROCm torch.', action: 'Install with --no-build-isolation after ROCm torch.', checked: '2026-10' },
  'liger-kernel': { status: 'ok', note: 'Pure Triton kernels; runs on ROCm.', action: 'No change.', checked: '2026-10' },
  unsloth: { status: 'partial', note: 'AMD support exists for Instinct and recent Radeon, with bitsandbytes ROCm underneath.', action: 'Follow the Unsloth AMD install path and pin bitsandbytes >= 0.48.', checked: '2026-10' },
  torchao: { status: 'partial', note: 'Core quantization works; some fp8 and sparse kernels are CUDA-only.', action: 'Check the kernel you use in the torchao ROCm CI.', checked: '2026-10' },
  'onnxruntime-gpu': { status: 'cuda-only', note: 'CUDA and TensorRT execution providers only.', action: 'Use onnxruntime-rocm (ROCm and MIGraphX execution providers).', replace: 'onnxruntime-rocm', checked: '2026-10' },
  tensorrt: { status: 'cuda-only', note: 'NVIDIA-only inference compiler.', action: 'Use MIGraphX (migraphx Python module ships with ROCm).', replace: '', checked: '2026-10' },
  'torch-tensorrt': { status: 'cuda-only', note: 'NVIDIA-only.', action: 'Use torch.compile, or MIGraphX via torch_migraphx.', replace: '', checked: '2026-10' },
  'cupy-cuda12x': { status: 'fork', note: 'CuPy has experimental ROCm support built from source.', action: 'Build CuPy with CUPY_INSTALL_USE_HIP=1 and HCC_AMDGPU_TARGET=<gfx>.', replace: '# cupy: build from source with CUPY_INSTALL_USE_HIP=1', checked: '2026-10' },
  'cupy-cuda11x': { status: 'fork', note: 'Same as cupy-cuda12x.', action: 'Build CuPy with CUPY_INSTALL_USE_HIP=1.', replace: '# cupy: build from source with CUPY_INSTALL_USE_HIP=1', checked: '2026-10' },
  pycuda: { status: 'cuda-only', note: 'Binds to the CUDA driver.', action: 'Use hip-python (pip install hip-python) for low-level HIP calls.', replace: 'hip-python', checked: '2026-10' },
  'nvidia-ml-py': { status: 'cuda-only', note: 'NVML bindings.', action: 'Use amdsmi (ships with ROCm) for GPU telemetry.', replace: '# amdsmi ships with ROCm (pip install /opt/rocm/share/amd_smi)', checked: '2026-10' },
  pynvml: { status: 'cuda-only', note: 'NVML bindings.', action: 'Use amdsmi.', replace: '# amdsmi ships with ROCm (pip install /opt/rocm/share/amd_smi)', checked: '2026-10' },
  gpustat: { status: 'cuda-only', note: 'Reads NVML.', action: 'Use amd-smi or rocm-smi on the command line.', replace: '', checked: '2026-10' },
  jax: { status: 'rocm', note: 'jax with the jax-rocm7-plugin wheels, or the rocm/jax Docker image.', action: 'Install jax plus the ROCm plugin instead of jax[cuda12].', checked: '2026-10' },
  tensorflow: { status: 'rocm', note: 'Published as tensorflow-rocm.', action: 'Replace with tensorflow-rocm of the matching version.', replace: 'tensorflow-rocm', checked: '2026-10' },
  numba: { status: 'partial', note: 'The numba.cuda target is NVIDIA-only; AMD maintains numba-hip.', action: 'CPU numba is fine; move @cuda.jit kernels to numba-hip or Triton.', checked: '2026-10' },
  cudf: { status: 'cuda-only', note: 'RAPIDS is CUDA-only.', action: 'Use pandas / polars on CPU, or keep this step on NVIDIA hardware.', replace: '', checked: '2026-10' },
  'cudf-cu12': { status: 'cuda-only', note: 'RAPIDS is CUDA-only.', action: 'Use polars on CPU.', replace: '', checked: '2026-10' },
  'cuml-cu12': { status: 'cuda-only', note: 'RAPIDS is CUDA-only.', action: 'Use scikit-learn on CPU.', replace: '', checked: '2026-10' },
  transformers: { status: 'ok', note: 'Pure Python on top of torch.', action: 'No change.', checked: '2026-10' },
  accelerate: { status: 'ok', note: 'Pure Python; detects ROCm torch as cuda devices.', action: 'No change.', checked: '2026-10' },
  peft: { status: 'ok', note: 'Pure Python.', action: 'No change.', checked: '2026-10' },
  diffusers: { status: 'ok', note: 'Pure Python.', action: 'No change.', checked: '2026-10' },
  datasets: { status: 'ok', note: 'Pure Python.', action: 'No change.', checked: '2026-10' },
  safetensors: { status: 'ok', note: 'CPU only.', action: 'No change.', checked: '2026-10' },
  numpy: { status: 'ok', note: 'CPU only.', action: 'No change.', checked: '2026-10' },
  // conda-only names
  'pytorch-cuda': { status: 'cuda-only', note: 'Conda metapackage that pulls CUDA runtime libraries.', action: 'Remove; install torch from the ROCm pip index instead.', replace: '', checked: '2026-10' },
  cudatoolkit: { status: 'cuda-only', note: 'CUDA runtime from conda.', action: 'Remove; ROCm comes from the system install or the rocm/pytorch image.', replace: '', checked: '2026-10' },
  'cuda-toolkit': { status: 'cuda-only', note: 'CUDA toolkit from conda.', action: 'Remove.', replace: '', checked: '2026-10' },
  'cuda-nvcc': { status: 'cuda-only', note: 'nvcc compiler.', action: 'Remove; hipcc comes with ROCm.', replace: '', checked: '2026-10' },
  cudnn: { status: 'cuda-only', note: 'cuDNN from conda.', action: 'Remove; MIOpen ships inside ROCm torch.', replace: '', checked: '2026-10' },
  pytorch: { status: 'rocm', note: 'Conda name for torch. The pytorch conda channel is discontinued.', action: `Move torch to pip from ${ROCM_INDEX}.`, checked: '2026-10' },
}

export interface Dep {
  name: string
  spec: string
  line: number
  raw: string
  source: 'pip' | 'conda'
  status: DepStatus
  note: string
  action: string
  checked?: string
  /** For torch-family pins with +cuXXX local versions */
  cudaLocal?: string
}

export interface DepsResult {
  kind: 'requirements' | 'conda'
  deps: Dep[]
  /** --index-url / --extra-index-url lines pointing at CUDA wheels */
  cudaIndexes: { line: number; url: string }[]
  /** conda channels that only make sense for CUDA */
  cudaChannels: string[]
  errors: { line: number; message: string }[]
}

export const normName = (n: string) => n.trim().toLowerCase().replace(/[_.]+/g, '-')

export function knownFor(name: string): Known | null {
  const n = normName(name)
  if (K[n]) return K[n]
  if (K[name.toLowerCase()]) return K[name.toLowerCase()]
  if (/^nvidia-/.test(n)) return { status: 'cuda-only', note: 'NVIDIA runtime library wheel (pulled in by CUDA torch).', action: 'Remove; ROCm torch bundles its own runtime libraries.', replace: '', checked: '2026-10' }
  if (/-cu1\d+$/.test(n)) return { status: 'cuda-only', note: 'Wheel built against a specific CUDA version.', action: 'Look for a -rocm variant or build from source.', replace: '', checked: '2026-10' }
  return null
}

function classify(name: string, spec: string, line: number, raw: string, source: Dep['source']): Dep {
  const k = knownFor(name)
  const local = spec.match(/\+(cu\d+)/)
  return {
    name, spec, line, raw, source,
    status: k?.status ?? 'unknown',
    note: k?.note ?? 'Not in the curated table. Pure-Python packages work unchanged; anything with compiled CUDA extensions needs a ROCm build.',
    action: k?.action ?? 'Check whether it ships compiled CUDA code (look for a cuda / cu12 extra or nvidia-* dependencies).',
    checked: k?.checked,
    cudaLocal: local?.[1],
  }
}

export function parseRequirements(text: string): DepsResult {
  const r: DepsResult = { kind: 'requirements', deps: [], cudaIndexes: [], cudaChannels: [], errors: [] }
  text.split(/\r?\n/).forEach((rawLine, i) => {
    const line = i + 1
    const s = rawLine.replace(/\s+#.*$/, '').replace(/^#.*$/, '').trim()
    if (!s) return
    const idx = s.match(/^(?:-i|--index-url|--extra-index-url|-f|--find-links)\s*=?\s*(\S+)/)
    if (idx) { if (/\/cu\d+|nvidia|cuda/i.test(idx[1])) r.cudaIndexes.push({ line, url: idx[1] }); return }
    if (/^-[rc]\s|^--requirement|^--constraint/.test(s)) { r.errors.push({ line, message: `Includes another file (${s}); upload it too to check its packages.` }); return }
    if (/^-/.test(s)) return
    const egg = s.match(/#egg=([\w.-]+)/) ?? s.match(/^([\w.-]+)\s*@\s*\S+/)
    if (egg) { r.deps.push(classify(egg[1], s.includes('@') ? '@ url' : 'git', line, rawLine, 'pip')); return }
    const m = s.match(/^([A-Za-z0-9][\w.-]*)\s*(\[[^\]]*\])?\s*([^;]*)/)
    if (!m) { r.errors.push({ line, message: `Could not read “${s}”` }); return }
    r.deps.push(classify(m[1], (m[2] ?? '') + m[3].trim(), line, rawLine, 'pip'))
  })
  return r
}

/** environment.yml: channels, conda dependencies, and the nested pip: list. Indentation-based, no YAML library. */
export function parseCondaEnv(text: string): DepsResult {
  const r: DepsResult = { kind: 'conda', deps: [], cudaIndexes: [], cudaChannels: [], errors: [] }
  let section = ''
  let pipIndent = -1
  text.split(/\r?\n/).forEach((rawLine, i) => {
    const line = i + 1
    const s = rawLine.replace(/\s+#.*$/, '')
    if (!s.trim() || s.trim().startsWith('#')) return
    const indent = s.length - s.trimStart().length
    const top = s.match(/^(\w[\w-]*)\s*:/)
    if (top && indent === 0) { section = top[1]; pipIndent = -1; return }
    const item = s.match(/^(\s*)-\s*(.+)$/)
    if (!item) return
    const v = item[2].trim()
    if (section === 'channels') { if (/nvidia|cuda/i.test(v)) r.cudaChannels.push(v); return }
    if (section !== 'dependencies') return
    if (pipIndent >= 0 && indent <= pipIndent) pipIndent = -1
    if (/^pip\s*:\s*$/.test(v)) { pipIndent = indent; return }
    if (pipIndent >= 0) {
      const sub = parseRequirements(v)
      for (const d of sub.deps) r.deps.push({ ...d, line, raw: rawLine, source: 'pip' })
      for (const x of sub.cudaIndexes) r.cudaIndexes.push({ ...x, line })
      return
    }
    const m = v.match(/^(?:[\w-]+::)?([A-Za-z0-9][\w.-]*)\s*(.*)$/)
    if (!m) { r.errors.push({ line, message: `Could not read “${v}”` }); return }
    r.deps.push(classify(m[1], m[2].trim(), line, rawLine, 'conda'))
  })
  return r
}

export const parseDeps = (name: string, text: string) => /\.ya?ml$/i.test(name) || /^\s*dependencies\s*:/m.test(text) ? parseCondaEnv(text) : parseRequirements(text)

/** A requirements file rewritten for ROCm: CUDA indexes swapped, CUDA-only packages removed or replaced, notes as comments. */
export function rocmRequirements(r: DepsResult): string {
  const out: string[] = [`# Rewritten for ROCm by HIP Port. Package table checked 2026-10; verify versions against your ROCm release.`, `--extra-index-url ${ROCM_INDEX}`]
  for (const d of r.deps) {
    if (d.source !== 'pip' && r.kind === 'conda' && ['pytorch', 'pytorch-cuda', 'cudatoolkit', 'cuda-toolkit', 'cuda-nvcc', 'cudnn'].includes(normName(d.name))) {
      if (normName(d.name) === 'pytorch') out.push(`torch${d.spec.replace(/^=+/, '==').replace(/\+cu\d+/, '')}`)
      else out.push(`# removed ${d.name}: ${d.action}`)
      continue
    }
    const k = knownFor(d.name)
    if (k && k.replace !== undefined) { out.push(k.replace ? k.replace : `# removed ${d.name}: ${k.action}`); continue }
    const spec = d.spec.replace(/\+cu\d+/, '').replace(/^=(?!=)/, '==')
    out.push(`${d.name}${spec}${d.status === 'partial' || d.status === 'fork' ? `  # ${d.action}` : ''}`)
  }
  return out.join('\n') + '\n'
}
