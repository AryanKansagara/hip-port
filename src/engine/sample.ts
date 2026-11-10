// Synthetic sample projects. Written for this tool; they are not taken from any real repository.
import type { SourceFile } from './scan.ts'

export interface Sample { id: string; name: string; blurb: string; target: string; files: SourceFile[] }

const HEADER = '// SYNTHETIC SAMPLE for HIP Port. Not from a real project.\n'

const reduce = `${HEADER}#include <cuda_runtime.h>
#include <cstdio>
#include "common.cuh"

#define WARP_SIZE 32

// Sum within a warp with shuffles.
__device__ float warpReduceSum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__global__ void blockSum(const float* __restrict__ x, float* out, int n) {
  __shared__ float partial[32];
  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp = tid / WARP_SIZE;
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) v += x[i];
  v = warpReduceSum(v);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < blockDim.x / WARP_SIZE ? partial[lane] : 0.f;
    v = warpReduceSum(v);
    if (lane == 0) atomicAdd(out, v);
  }
}

// Count positive values per warp with a ballot.
__global__ void countPositive(const float* x, int* counts, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned mask = __ballot_sync(0xffffffff, i < n && x[i] > 0.f);
  if ((threadIdx.x & 31) == 0) atomicAdd(counts, __popc(mask));
}

// Legacy shared-memory tail that relies on lockstep warps.
__global__ void legacyReduce(const float* x, float* out) {
  extern __shared__ float s[];
  unsigned tid = threadIdx.x;
  s[tid] = x[blockIdx.x * blockDim.x + tid];
  __syncthreads();
  for (unsigned st = blockDim.x / 2; st > 32; st >>= 1) {
    if (tid < st) s[tid] += s[tid + st];
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* vs = s;
    vs[tid] += vs[tid + 32]; vs[tid] += vs[tid + 16]; vs[tid] += vs[tid + 8];
    vs[tid] += vs[tid + 4];  vs[tid] += vs[tid + 2];  vs[tid] += vs[tid + 1];
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

float sum(const float* d_x, int n) {
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  blockSum<<<256, 512>>>(d_x, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  float h = 0.f;
  CUDA_CHECK(cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  return h;
}
`

const common = `${HEADER}#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                         \\
  cudaError_t err_ = (call);                                          \\
  if (err_ != cudaSuccess) {                                          \\
    fprintf(stderr, "CUDA error %s at %s:%d\\n",                      \\
            cudaGetErrorString(err_), __FILE__, __LINE__);            \\
    std::exit(1);                                                     \\
  }                                                                   \\
} while (0)

// Lane id via PTX special register.
__device__ __forceinline__ unsigned laneId() {
  unsigned id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ unsigned laneMaskLt() {
  return (1u << laneId()) - 1;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define HAS_ASYNC_COPY 1
#endif
`

const gemm = `${HEADER}#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.cuh"

// C = alpha * A * B + beta * C, column-major, with timing.
float gemm(int m, int n, int k, const float* dA, const float* dB, float* dC, cudaStream_t stream) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  const float alpha = 1.f, beta = 0.f;

  cudaEventRecord(start, stream);
  cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                  &alpha, dA, m, dB, k, &beta, dC, m);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  if (st != CUBLAS_STATUS_SUCCESS) fprintf(stderr, "cublasSgemm failed\\n");

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  return ms;
}

// Mixed precision path: fp16 inputs, fp32 accumulate.
void gemmHalf(cublasHandle_t h, int m, int n, int k, const void* A, const void* B, void* C) {
  const float alpha = 1.f, beta = 0.f;
  cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
               A, CUDA_R_16F, m, B, CUDA_R_16F, n, &beta, C, CUDA_R_32F, m,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
`

const texture = `${HEADER}#include <cuda_runtime.h>
#include "common.cuh"

// 2D box blur through a texture reference (legacy API).
texture<float, 2, cudaReadModeElementType> texImg;

__global__ void boxBlur(float* out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float acc = 0.f;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      acc += tex2D(texImg, x + dx + 0.5f, y + dy + 0.5f);
  out[y * w + x] = acc / 9.f;
}

void blur(const float* dImg, size_t pitch, float* dOut, int w, int h) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  texImg.filterMode = cudaFilterModePoint;
  CUDA_CHECK(cudaBindTexture2D(nullptr, texImg, dImg, desc, w, h, pitch));
  dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
  boxBlur<<<grid, block>>>(dOut, w, h);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaUnbindTexture(texImg);
}
`

const softmax = `${HEADER}#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "common.cuh"

namespace cg = cooperative_groups;

// Row softmax: one block per row, CUB block reduce for max and sum.
template <int BLOCK>
__global__ void rowSoftmax(const float* x, float* y, int cols) {
  using BlockReduce = cub::BlockReduce<float, BLOCK>;
  __shared__ typename BlockReduce::TempStorage tmp;
  __shared__ float rowMax, rowSum;
  const float* xr = x + blockIdx.x * cols;
  float m = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += BLOCK) m = fmaxf(m, xr[c]);
  m = BlockReduce(tmp).Reduce(m, cub::Max());
  if (threadIdx.x == 0) rowMax = m;
  __syncthreads();
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += BLOCK) s += __expf(xr[c] - rowMax);
  s = BlockReduce(tmp).Sum(s);
  if (threadIdx.x == 0) rowSum = s;
  __syncthreads();
  for (int c = threadIdx.x; c < cols; c += BLOCK) y[blockIdx.x * cols + c] = __expf(xr[c] - rowMax) / rowSum;
}

// Grid-wide normalisation with a cooperative grid sync.
__global__ void normalizeGrid(float* v, float* total, int n) {
  cg::grid_group grid = cg::this_grid();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cg::this_thread_block());
  float s = 0.f;
  for (int i = grid.thread_rank(); i < n; i += grid.size()) s += v[i];
  for (int o = tile.size() / 2; o > 0; o /= 2) s += tile.shfl_down(s, o);
  if (tile.thread_rank() == 0) atomicAdd(total, s);
  grid.sync();
  for (int i = grid.thread_rank(); i < n; i += grid.size()) v[i] /= *total;
}

void topk(thrust::device_vector<float>& scores) {
  thrust::sort(scores.begin(), scores.end(), thrust::greater<float>());
}

void run(const float* x, float* y, int rows, int cols) {
  rowSoftmax<256><<<rows, 256, 0, 0>>>(x, y, cols);
  CUDA_CHECK(cudaGetLastError());
}
`

const requirements = `# SYNTHETIC SAMPLE for HIP Port: a typical CUDA training/serving stack.
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.3.1+cu121
torchvision==0.18.1+cu121
transformers>=4.44
accelerate
peft
bitsandbytes==0.43.1
flash-attn==2.5.9.post1
xformers==0.0.27
vllm==0.5.3
triton==2.3.1
nvidia-ml-py
cupy-cuda12x
onnxruntime-gpu
deepspeed
einops
`

// Second sample: LLM inference kernels with fp8, cuRAND and a conda environment.
const quant = `${HEADER}#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

// Per-tensor fp8 quantisation of bf16 activations.
__global__ void quantFp8(const __nv_bfloat16* in, __nv_fp8_e4m3* out, const float* scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __nv_fp8_e4m3(__bfloat162float(in[i]) / *scale);
}

// absmax with an sm_80 warp reduce intrinsic.
__global__ void absmax(const __nv_bfloat16* in, unsigned* outBits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float a = i < n ? fabsf(__bfloat162float(in[i])) : 0.f;
  unsigned bits = __float_as_uint(a);
  bits = __reduce_max_sync(0xffffffff, bits);
  if ((threadIdx.x & 31) == 0) atomicMax(outBits, bits);
}

void launchQuant(const __nv_bfloat16* in, __nv_fp8_e4m3* out, const float* scale, int n, cudaStream_t s) {
  quantFp8<<<(n + 255) / 256, 256, 0, s>>>(in, out, scale, n);
}
`

const sampling = `${HEADER}#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <nvToolsExt.h>

__global__ void initRng(curandState* st, unsigned long long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, i, 0, &st[i]);
}

// Gumbel-max sampling over one row per warp.
__global__ void sampleRows(curandState* st, const float* logits, int* tokens, int vocab) {
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  curandState s = st[row * 32 + lane];
  float best = -1e30f; int arg = 0;
  for (int v = lane; v < vocab; v += 32) {
    float g = logits[row * vocab + v] - logf(-logf(curand_uniform(&s)));
    if (g > best) { best = g; arg = v; }
  }
  for (int o = 16; o > 0; o >>= 1) {
    float ob = __shfl_xor_sync(0xffffffff, best, o);
    int oa = __shfl_xor_sync(0xffffffff, arg, o);
    if (ob > best) { best = ob; arg = oa; }
  }
  if (lane == 0) tokens[row] = arg;
  st[row * 32 + lane] = s;
}

void sample(curandState* st, const float* logits, int* tokens, int rows, int vocab, cudaStream_t s) {
  nvtxRangePushA("sample");
  sampleRows<<<(rows + 7) / 8, 256, 0, s>>>(st, logits, tokens, vocab);
  nvtxRangePop();
}
`

const envYml = `# SYNTHETIC SAMPLE for HIP Port: conda environment for an inference service.
name: serve
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch=2.4.0
  - pytorch-cuda=12.1
  - cudatoolkit=12.1
  - numpy
  - pip
  - pip:
      - transformers==4.45.2
      - flash-attn==2.6.3
      - autoawq==0.2.6
      - flashinfer
      - pynvml
      - sglang==0.3.2
`

export const SAMPLES: Sample[] = [
  {
    id: 'mini-repo', name: 'Synthetic CUDA mini-repo', target: 'gfx942',
    blurb: 'Warp-shuffle reductions, a cuBLAS GEMM, a texture-reference blur, inline PTX, cooperative groups and a CUDA 12.1 requirements file.',
    files: [
      { path: 'src/common.cuh', text: common },
      { path: 'src/reduce.cu', text: reduce },
      { path: 'src/gemm.cu', text: gemm },
      { path: 'src/texture_blur.cu', text: texture },
      { path: 'src/softmax.cu', text: softmax },
      { path: 'requirements.txt', text: requirements },
    ],
  },
  {
    id: 'llm-kernels', name: 'Synthetic LLM serving kernels', target: 'gfx1201',
    blurb: 'FP8 quantisation, an sm_80 warp-reduce intrinsic, Gumbel sampling with cuRAND and NVTX, and a conda environment with flash-attn and AWQ.',
    files: [
      { path: 'kernels/quant_fp8.cu', text: quant },
      { path: 'kernels/sampling.cu', text: sampling },
      { path: 'environment.yml', text: envYml },
    ],
  },
]

/** The kernel shown in the landing-page hero. */
export const HERO_KERNEL = `#include <cuda_runtime.h>
#define WARP_SIZE 32

__device__ float warpSum(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

__global__ void dot(const float* a, const float* b,
                    float* out, int n) {
  int lane = threadIdx.x % 32;
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x)
    v += a[i] * b[i];
  v = warpSum(v);
  unsigned live = __ballot_sync(0xffffffff, v != 0.f);
  if (lane == 0 && live) atomicAdd(out, v);
}

void run(const float* a, const float* b, float* out,
         int n, cudaStream_t s) {
  cudaMemsetAsync(out, 0, sizeof(float), s);
  dot<<<256, 256, 0, s>>>(a, b, out, n);
  cudaStreamSynchronize(s);
}
`
