// CUDA -> HIP identifier and header mapping.
// Three layers, first hit wins:
//   1. the curated table below: notes, review status and the parts hipify does not cover
//      (cuDNN -> MIOpen, NVTX -> ROCTX, WMMA, version macros);
//   2. the official hipify-perl tables (hipifyTable.ts, generated): renames, the list of CUDA
//      identifiers hipify reports as unsupported, and CUDA deprecation / removal versions;
//   3. two family rules where no rename happens: cuDNN (port by hand) and NCCL (RCCL keeps names).
// Anything else that looks like CUDA is reported as unmapped and left untouched.
import { RENAMES, EXPERIMENTAL, UNSUPPORTED, DEVICE_UNSUPPORTED, DEPRECATED, REMOVED, INCLUDES } from './hipifyTable.ts'

export type Lib = 'runtime' | 'driver' | 'cublas' | 'cublaslt' | 'cufft' | 'curand' | 'cusparse' | 'cusolver' | 'cutensor' | 'nvrtc' | 'cufile' | 'cudnn' | 'nccl' | 'thrust' | 'cub' | 'nvtx' | 'complex' | 'fp' | 'wmma' | 'coop' | 'macro'
export type Status = 'ok' | 'partial' | 'deprecated' | 'unsupported'

export interface Mapping {
  hip: string | null
  lib: Lib
  status: Status
  note?: string
  /** 'table' = curated entry, 'hipify' = official hipify-perl table, 'family' = prefix rule */
  via: 'table' | 'hipify' | 'family'
}

export const LIBS: Record<Lib, { cuda: string; rocm: string; note?: string }> = {
  runtime: { cuda: 'CUDA Runtime', rocm: 'HIP Runtime' },
  driver: { cuda: 'CUDA Driver API', rocm: 'HIP Module API', note: 'Contexts are deprecated in HIP; prefer devices and streams.' },
  cublas: { cuda: 'cuBLAS', rocm: 'hipBLAS (rocBLAS backend)' },
  cublaslt: { cuda: 'cuBLASLt', rocm: 'hipBLASLt', note: 'Epilogues and data types vary by gfx target; check the hipBLASLt support matrix.' },
  cufft: { cuda: 'cuFFT', rocm: 'hipFFT (rocFFT backend)' },
  curand: { cuda: 'cuRAND', rocm: 'hipRAND (rocRAND backend)' },
  cusparse: { cuda: 'cuSPARSE', rocm: 'hipSPARSE (rocSPARSE backend)' },
  cusolver: { cuda: 'cuSOLVER', rocm: 'hipSOLVER (rocSOLVER backend)' },
  cutensor: { cuda: 'cuTENSOR', rocm: 'hipTensor', note: 'hipTensor covers contraction, reduction and permutation; check the operation you use.' },
  nvrtc: { cuda: 'NVRTC', rocm: 'hipRTC', note: 'hipRTC compiles HIP source to a code object for the current device; PTX and NVVM outputs do not exist.' },
  cufile: { cuda: 'cuFile (GDS)', rocm: 'hipFile' },
  cudnn: { cuda: 'cuDNN', rocm: 'MIOpen', note: 'Not a drop-in: MIOpen needs Find/solution calls before convolutions, and descriptor setters differ.' },
  nccl: { cuda: 'NCCL', rocm: 'RCCL', note: 'RCCL keeps the nccl* API names; only the header and link library change.' },
  thrust: { cuda: 'Thrust', rocm: 'rocThrust', note: 'Same thrust:: namespace and headers; link rocThrust.' },
  cub: { cuda: 'CUB', rocm: 'hipCUB', note: 'cub:: becomes hipcub::. Block and warp primitives assume the device warp size.' },
  nvtx: { cuda: 'NVTX', rocm: 'ROCTX (roctracer)' },
  complex: { cuda: 'cuComplex', rocm: 'hip_complex' },
  fp: { cuda: 'CUDA fp16/bf16/fp8 types', rocm: 'HIP fp16/bf16/fp8 types' },
  wmma: { cuda: 'WMMA (mma.h)', rocm: 'rocWMMA', note: 'Different fragment shapes per architecture; port by hand.' },
  coop: { cuda: 'Cooperative groups', rocm: 'HIP cooperative groups' },
  macro: { cuda: 'Compiler macros', rocm: 'HIP compiler macros' },
}

type Row = [hip: string | null, lib: Lib, status?: Status, note?: string]
const T: Record<string, Row> = {
  // Runtime exceptions and notes (everything else in cuda* follows the runtime family rule)
  cudaThreadSynchronize: ['hipDeviceSynchronize', 'runtime', 'ok', 'cudaThreadSynchronize is a deprecated alias.'],
  cudaThreadExit: ['hipDeviceReset', 'runtime', 'ok', 'cudaThreadExit is a deprecated alias.'],
  cudaErrorMemoryAllocation: ['hipErrorOutOfMemory', 'runtime'],
  cudaErrorInitializationError: ['hipErrorNotInitialized', 'runtime'],
  cudaErrorInvalidDevicePointer: ['hipErrorInvalidDevicePointer', 'runtime'],
  cudaMallocManaged: ['hipMallocManaged', 'runtime', 'partial', 'Page migration on Instinct GPUs needs HSA_XNACK=1 and an XNACK-enabled build (gfx90a:xnack+); otherwise managed memory behaves like pinned host memory.'],
  cudaMemAdvise: ['hipMemAdvise', 'runtime', 'partial', 'Advice hints are honoured only with XNACK enabled.'],
  cudaMemPrefetchAsync: ['hipMemPrefetchAsync', 'runtime', 'partial', 'Requires XNACK for real migration.'],
  cudaMemcpyToSymbol: ['hipMemcpyToSymbol', 'runtime', 'ok', 'The symbol argument is wrapped in HIP_SYMBOL() for you.'],
  cudaMemcpyFromSymbol: ['hipMemcpyFromSymbol', 'runtime', 'ok', 'The symbol argument is wrapped in HIP_SYMBOL() for you.'],
  cudaGetSymbolAddress: ['hipGetSymbolAddress', 'runtime', 'ok', 'The symbol argument is wrapped in HIP_SYMBOL() for you.'],
  cudaDeviceSetSharedMemConfig: ['hipDeviceSetSharedMemConfig', 'runtime', 'partial', 'Accepted but ignored: AMD LDS has no bank-size setting.'],
  cudaFuncSetSharedMemConfig: ['hipFuncSetSharedMemConfig', 'runtime', 'partial', 'Accepted but ignored on AMD GPUs.'],
  cudaDeviceSetCacheConfig: ['hipDeviceSetCacheConfig', 'runtime', 'partial', 'Accepted but ignored on AMD GPUs.'],
  cudaFuncSetCacheConfig: ['hipFuncSetCacheConfig', 'runtime', 'partial', 'Accepted but ignored on AMD GPUs.'],
  cudaBindTexture: ['hipBindTexture', 'runtime', 'deprecated', 'Texture references are deprecated in HIP and unsupported in recent ROCm. Rewrite with texture objects.'],
  cudaBindTexture2D: ['hipBindTexture2D', 'runtime', 'deprecated', 'Texture references are deprecated in HIP and unsupported in recent ROCm. Rewrite with texture objects.'],
  cudaBindTextureToArray: ['hipBindTextureToArray', 'runtime', 'deprecated', 'Texture references are deprecated. Rewrite with texture objects.'],
  cudaUnbindTexture: ['hipUnbindTexture', 'runtime', 'deprecated', 'Texture references are deprecated. Rewrite with texture objects.'],
  cudaGetTextureReference: ['hipGetTextureReference', 'runtime', 'deprecated', 'Texture references are deprecated; use hipTextureObject_t.'],
  cudaSetDoubleForDevice: [null, 'runtime', 'unsupported', 'Removed from CUDA and absent from HIP; delete the call.'],
  cudaSetDoubleForHost: [null, 'runtime', 'unsupported', 'Removed from CUDA and absent from HIP; delete the call.'],
  cudaProfilerInitialize: [null, 'runtime', 'unsupported', 'Use rocprofv3 on the command line instead.'],
  cudaProfilerStart: ['hipProfilerStart', 'runtime', 'deprecated', 'Prefer rocprofv3 range markers (roctx).'],
  cudaProfilerStop: ['hipProfilerStop', 'runtime', 'deprecated', 'Prefer rocprofv3 range markers (roctx).'],
  cudaGLRegisterBufferObject: [null, 'runtime', 'unsupported', 'Legacy GL interop; use hipGraphicsGLRegisterBuffer.'],
  cudaLaunchKernelEx: [null, 'runtime', 'unsupported', 'The C++ template launch is not in HIP; hipify only maps cudaLaunchKernelExC to hipLaunchKernelExC.'],
  cudaLaunchKernelExC: ['hipLaunchKernelExC', 'runtime', 'partial', 'Launch attributes such as clusters are NVIDIA-only.'],
  cudaFuncAttributeNonPortableClusterSizeAllowed: ['hipFuncAttributeNonPortableClusterSizeAllowed', 'runtime', 'partial', 'The name exists in HIP but thread block clusters are NVIDIA Hopper-only.'],
  cudaDevAttrComputeCapabilityMajor: ['hipDeviceAttributeComputeCapabilityMajor', 'runtime', 'partial', 'AMD reports a pseudo compute capability; branch on gcnArchName instead.'],
  cudaDevAttrComputeCapabilityMinor: ['hipDeviceAttributeComputeCapabilityMinor', 'runtime', 'partial', 'AMD reports a pseudo compute capability; branch on gcnArchName instead.'],
  cudaDevAttrMultiProcessorCount: ['hipDeviceAttributeMultiprocessorCount', 'runtime'],
  cudaDevAttrMaxSharedMemoryPerBlockOptin: ['hipDeviceAttributeSharedMemPerBlockOptin', 'runtime'],
  cudaDevAttrWarpSize: ['hipDeviceAttributeWarpSize', 'runtime'],
  cudaDevAttrMaxThreadsPerBlock: ['hipDeviceAttributeMaxThreadsPerBlock', 'runtime'],
  cudaDevAttrL2CacheSize: ['hipDeviceAttributeL2CacheSize', 'runtime'],
  cudaDeviceAttr: ['hipDeviceAttribute_t', 'runtime'],
  cudaDataType: ['hipDataType', 'runtime'],
  cudaDataType_t: ['hipDataType', 'runtime'],
  CUDA_R_16F: ['HIP_R_16F', 'runtime'], CUDA_R_16BF: ['HIP_R_16BF', 'runtime'], CUDA_R_32F: ['HIP_R_32F', 'runtime'], CUDA_R_64F: ['HIP_R_64F', 'runtime'],
  CUDA_R_8I: ['HIP_R_8I', 'runtime'], CUDA_R_32I: ['HIP_R_32I', 'runtime'], CUDA_C_32F: ['HIP_C_32F', 'runtime'], CUDA_C_64F: ['HIP_C_64F', 'runtime'],
  CUDA_R_8F_E4M3: ['HIP_R_8F_E4M3', 'runtime', 'partial', 'gfx942 (MI300) implements the FNUZ variant (HIP_R_8F_E4M3_FNUZ); gfx950 and RDNA4 use OCP E4M3.'],
  CUDA_R_8F_E5M2: ['HIP_R_8F_E5M2', 'runtime', 'partial', 'gfx942 (MI300) implements the FNUZ variant (HIP_R_8F_E5M2_FNUZ).'],
  // Driver API
  cuInit: ['hipInit', 'driver'],
  cuDriverGetVersion: ['hipDriverGetVersion', 'driver'],
  cuDeviceGet: ['hipDeviceGet', 'driver'],
  cuDeviceGetCount: ['hipGetDeviceCount', 'driver'],
  cuDeviceGetName: ['hipDeviceGetName', 'driver'],
  cuDeviceGetAttribute: ['hipDeviceGetAttribute', 'driver'],
  cuDeviceTotalMem: ['hipDeviceTotalMem', 'driver'],
  cuCtxCreate: ['hipCtxCreate', 'driver', 'deprecated', 'HIP contexts are deprecated; use hipSetDevice and the primary context.'],
  cuCtxDestroy: ['hipCtxDestroy', 'driver', 'deprecated', 'HIP contexts are deprecated.'],
  cuCtxSetCurrent: ['hipCtxSetCurrent', 'driver', 'deprecated', 'HIP contexts are deprecated; use hipSetDevice.'],
  cuCtxGetCurrent: ['hipCtxGetCurrent', 'driver', 'deprecated', 'HIP contexts are deprecated.'],
  cuCtxSynchronize: ['hipCtxSynchronize', 'driver', 'deprecated', 'Use hipDeviceSynchronize.'],
  cuDevicePrimaryCtxRetain: ['hipDevicePrimaryCtxRetain', 'driver'],
  cuDevicePrimaryCtxRelease: ['hipDevicePrimaryCtxRelease', 'driver'],
  cuModuleLoad: ['hipModuleLoad', 'driver', 'partial', 'Load a code object (.hsaco or .co) built for your gfx target, not PTX or cubin.'],
  cuModuleLoadData: ['hipModuleLoadData', 'driver', 'partial', 'Pass a code object built for your gfx target, not PTX.'],
  cuModuleLoadDataEx: ['hipModuleLoadDataEx', 'driver', 'partial', 'JIT options for PTX do not apply.'],
  cuModuleGetFunction: ['hipModuleGetFunction', 'driver'],
  cuModuleGetGlobal: ['hipModuleGetGlobal', 'driver'],
  cuModuleUnload: ['hipModuleUnload', 'driver'],
  cuLaunchKernel: ['hipModuleLaunchKernel', 'driver'],
  cuMemAlloc: ['hipMalloc', 'driver'],
  cuMemAlloc_v2: ['hipMalloc', 'driver'],
  cuMemFree: ['hipFree', 'driver'],
  cuMemFree_v2: ['hipFree', 'driver'],
  cuMemcpyHtoD: ['hipMemcpyHtoD', 'driver'],
  cuMemcpyDtoH: ['hipMemcpyDtoH', 'driver'],
  cuMemcpyDtoD: ['hipMemcpyDtoD', 'driver'],
  cuMemcpyHtoDAsync: ['hipMemcpyHtoDAsync', 'driver'],
  cuMemcpyDtoHAsync: ['hipMemcpyDtoHAsync', 'driver'],
  cuMemsetD32: ['hipMemsetD32', 'driver'],
  cuStreamCreate: ['hipStreamCreateWithFlags', 'driver', 'partial', 'Argument order matches hipStreamCreateWithFlags(&s, flags).'],
  cuStreamSynchronize: ['hipStreamSynchronize', 'driver'],
  cuStreamDestroy: ['hipStreamDestroy', 'driver'],
  cuEventCreate: ['hipEventCreateWithFlags', 'driver'],
  cuEventRecord: ['hipEventRecord', 'driver'],
  cuEventElapsedTime: ['hipEventElapsedTime', 'driver'],
  cuGetErrorString: ['hipDrvGetErrorString', 'driver'],
  cuGetErrorName: ['hipDrvGetErrorName', 'driver'],
  cuLinkCreate: ['hiprtcLinkCreate', 'driver', 'partial', 'Linking goes through hipRTC with LLVM bitcode, not PTX.'],
  CUdevice: ['hipDevice_t', 'driver'],
  CUcontext: ['hipCtx_t', 'driver', 'deprecated', 'HIP contexts are deprecated.'],
  CUmodule: ['hipModule_t', 'driver'],
  CUfunction: ['hipFunction_t', 'driver'],
  CUstream: ['hipStream_t', 'driver'],
  CUevent: ['hipEvent_t', 'driver'],
  CUdeviceptr: ['hipDeviceptr_t', 'driver'],
  CUresult: ['hipError_t', 'driver'],
  CUDA_SUCCESS: ['hipSuccess', 'driver'],
  CUDA_ERROR_OUT_OF_MEMORY: ['hipErrorOutOfMemory', 'driver'],
  CUDA_ERROR_INVALID_VALUE: ['hipErrorInvalidValue', 'driver'],
  CUDA_ERROR_NOT_INITIALIZED: ['hipErrorNotInitialized', 'driver'],
  CU_STREAM_NON_BLOCKING: ['hipStreamNonBlocking', 'driver'],
  CU_STREAM_DEFAULT: ['hipStreamDefault', 'driver'],
  CU_EVENT_DEFAULT: ['hipEventDefault', 'driver'],
  CU_EVENT_DISABLE_TIMING: ['hipEventDisableTiming', 'driver'],
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: ['hipDeviceAttributeMultiprocessorCount', 'driver'],
  CU_DEVICE_ATTRIBUTE_WARP_SIZE: ['hipDeviceAttributeWarpSize', 'driver'],
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: ['hipDeviceAttributeMaxThreadsPerBlock', 'driver'],
  // cuBLAS specifics
  cublasSetMathMode: ['hipblasSetMathMode', 'cublas', 'partial', 'Math modes are hints on AMD; tensor-op selection is automatic in rocBLAS/hipBLASLt.'],
  CUBLAS_TENSOR_OP_MATH: ['HIPBLAS_TENSOR_OP_MATH', 'cublas', 'partial', 'Hint only on AMD.'],
  CUBLAS_DEFAULT_MATH: ['HIPBLAS_DEFAULT_MATH', 'cublas'],
  CUBLAS_GEMM_DEFAULT_TENSOR_OP: [null, 'cublas', 'unsupported', 'No tensor-op algorithm enum in hipBLAS; use HIPBLAS_GEMM_DEFAULT.'],
  cublasGemmEx: ['hipblasGemmEx', 'cublas', 'partial', 'Since hipBLAS 2.0 the compute type is hipblasComputeType_t (HIPBLAS_COMPUTE_32F), not a data type. Build with HIPBLAS_V2 on older ROCm.'],
  cublasGemmStridedBatchedEx: ['hipblasGemmStridedBatchedEx', 'cublas', 'partial', 'Compute type is hipblasComputeType_t since hipBLAS 2.0.'],
  cublasLtMatmul: ['hipblasLtMatmul', 'cublaslt', 'partial', 'Check the hipBLASLt epilogue and type support for your gfx target.'],
  // cuDNN -> MIOpen: only the handle and descriptor plumbing is mechanical
  cudnnHandle_t: ['miopenHandle_t', 'cudnn'],
  cudnnCreate: ['miopenCreate', 'cudnn'],
  cudnnDestroy: ['miopenDestroy', 'cudnn'],
  cudnnSetStream: ['miopenSetStream', 'cudnn'],
  cudnnStatus_t: ['miopenStatus_t', 'cudnn'],
  CUDNN_STATUS_SUCCESS: ['miopenStatusSuccess', 'cudnn'],
  cudnnGetErrorString: ['miopenGetErrorString', 'cudnn'],
  cudnnTensorDescriptor_t: ['miopenTensorDescriptor_t', 'cudnn'],
  cudnnCreateTensorDescriptor: ['miopenCreateTensorDescriptor', 'cudnn'],
  cudnnDestroyTensorDescriptor: ['miopenDestroyTensorDescriptor', 'cudnn'],
  cudnnSetTensor4dDescriptor: ['miopenSet4dTensorDescriptor', 'cudnn', 'partial', 'MIOpen takes (desc, dataType, n, c, h, w) and has no format argument; NCHW is assumed.'],
  cudnnFilterDescriptor_t: ['miopenTensorDescriptor_t', 'cudnn', 'partial', 'MIOpen describes filters with a tensor descriptor.'],
  cudnnConvolutionDescriptor_t: ['miopenConvolutionDescriptor_t', 'cudnn'],
  cudnnCreateConvolutionDescriptor: ['miopenCreateConvolutionDescriptor', 'cudnn'],
  cudnnConvolutionForward: ['miopenConvolutionForward', 'cudnn', 'partial', 'Call miopenFindConvolutionForwardAlgorithm first; the algorithm enum and workspace query differ.'],
  cudnnActivationForward: ['miopenActivationForward', 'cudnn', 'partial', 'Activation descriptors are created with miopenSetActivationDescriptor (mode, alpha, beta, gamma).'],
  cudnnSoftmaxForward: ['miopenSoftmaxForward_V2', 'cudnn', 'partial', 'Algorithm and mode enums differ.'],
  cudnnPoolingForward: ['miopenPoolingForward', 'cudnn', 'partial', 'MIOpen pooling takes a workspace argument for the backward pass.'],
  CUDNN_DATA_FLOAT: ['miopenFloat', 'cudnn'],
  CUDNN_DATA_HALF: ['miopenHalf', 'cudnn'],
  CUDNN_DATA_BFLOAT16: ['miopenBFloat16', 'cudnn'],
  CUDNN_TENSOR_NCHW: [null, 'cudnn', 'partial', 'No layout argument in MIOpen 4d descriptors; delete it.'],
  // NCCL -> RCCL keeps names, only notes
  // NVTX -> ROCTX
  nvtxRangePushA: ['roctxRangePushA', 'nvtx'],
  nvtxRangePush: ['roctxRangePushA', 'nvtx'],
  nvtxRangePop: ['roctxRangePop', 'nvtx'],
  nvtxMarkA: ['roctxMarkA', 'nvtx'],
  nvtxRangeStartA: ['roctxRangeStartA', 'nvtx'],
  nvtxRangeEnd: ['roctxRangeStop', 'nvtx'],
  // cuComplex
  cuFloatComplex: ['hipFloatComplex', 'complex'], cuDoubleComplex: ['hipDoubleComplex', 'complex'], cuComplex: ['hipComplex', 'complex'],
  make_cuFloatComplex: ['make_hipFloatComplex', 'complex'], make_cuDoubleComplex: ['make_hipDoubleComplex', 'complex'], make_cuComplex: ['make_hipComplex', 'complex'],
  cuCrealf: ['hipCrealf', 'complex'], cuCimagf: ['hipCimagf', 'complex'], cuCaddf: ['hipCaddf', 'complex'], cuCmulf: ['hipCmulf', 'complex'], cuConjf: ['hipConjf', 'complex'], cuCabsf: ['hipCabsf', 'complex'],
  cuCreal: ['hipCreal', 'complex'], cuCimag: ['hipCimag', 'complex'], cuCadd: ['hipCadd', 'complex'], cuCmul: ['hipCmul', 'complex'],
  // Low-precision types
  __nv_bfloat16: ['__hip_bfloat16', 'fp'],
  __nv_bfloat162: ['__hip_bfloat162', 'fp'],
  nv_bfloat16: ['hip_bfloat16', 'fp'],
  __nv_fp8_e4m3: ['__hip_fp8_e4m3_fnuz', 'fp', 'partial', 'hipify maps to the FNUZ type (gfx942, MI300): different bias and NaN encoding from CUDA. On gfx950 and RDNA4 use the OCP type __hip_fp8_e4m3.'],
  __nv_fp8_e5m2: ['__hip_fp8_e5m2_fnuz', 'fp', 'partial', 'hipify maps to the FNUZ type (gfx942, MI300). On gfx950 and RDNA4 use the OCP type __hip_fp8_e5m2.'],
  // WMMA
  nvcuda: ['rocwmma', 'wmma', 'partial', 'rocWMMA mirrors the wmma API but fragment sizes and supported types depend on the gfx target (MFMA on CDNA, WMMA on RDNA3+).'],
  // Macros
  __CUDACC__: ['__HIPCC__', 'macro'],
  __NVCC__: ['__HIPCC__', 'macro', 'partial', 'hipcc defines __HIPCC__, not __NVCC__.'],
  CUDART_VERSION: ['HIP_VERSION', 'macro', 'partial', 'Different numbering: HIP_VERSION is major*10000000 + minor*100000 + patch.'],
  CUDA_VERSION: ['HIP_VERSION', 'macro', 'partial', 'Different numbering scheme.'],
  // Cooperative groups stay cooperative_groups; only the header changes.
}

export const HEADERS: Record<string, [string | null, Lib, Status?, string?]> = {
  'cuda_runtime.h': ['hip/hip_runtime.h', 'runtime'],
  'cuda_runtime_api.h': ['hip/hip_runtime_api.h', 'runtime'],
  'cuda.h': ['hip/hip_runtime.h', 'driver'],
  'device_launch_parameters.h': ['hip/hip_runtime.h', 'runtime'],
  'cuda_profiler_api.h': ['hip/hip_runtime_api.h', 'runtime'],
  'cuda_fp16.h': ['hip/hip_fp16.h', 'fp'],
  'cuda_bf16.h': ['hip/hip_bf16.h', 'fp'],
  'cuda_fp8.h': ['hip/hip_fp8.h', 'fp', 'partial', 'FP8 encodings differ between gfx942 (FNUZ) and gfx950 / RDNA4 (OCP).'],
  'cuComplex.h': ['hip/hip_complex.h', 'complex'],
  'math_constants.h': ['hip/hip_math_constants.h', 'runtime'],
  'cublas_v2.h': ['hipblas/hipblas.h', 'cublas'],
  'cublas.h': ['hipblas/hipblas.h', 'cublas'],
  'cublasLt.h': ['hipblaslt/hipblaslt.h', 'cublaslt'],
  'cufft.h': ['hipfft/hipfft.h', 'cufft'],
  'cufftXt.h': ['hipfft/hipfftXt.h', 'cufft'],
  'curand.h': ['hiprand/hiprand.h', 'curand'],
  'curand_kernel.h': ['hiprand/hiprand_kernel.h', 'curand'],
  'cusparse.h': ['hipsparse/hipsparse.h', 'cusparse'],
  'cusparse_v2.h': ['hipsparse/hipsparse.h', 'cusparse'],
  'cusolverDn.h': ['hipsolver/hipsolver.h', 'cusolver'],
  'cudnn.h': ['miopen/miopen.h', 'cudnn', 'partial', 'MIOpen is not API compatible with cuDNN.'],
  'nccl.h': ['rccl/rccl.h', 'nccl'],
  'cub/cub.cuh': ['hipcub/hipcub.hpp', 'cub'],
  'cub/block/block_reduce.cuh': ['hipcub/hipcub.hpp', 'cub'],
  'cub/warp/warp_reduce.cuh': ['hipcub/hipcub.hpp', 'cub'],
  'cub/device/device_radix_sort.cuh': ['hipcub/hipcub.hpp', 'cub'],
  'cooperative_groups.h': ['hip/hip_cooperative_groups.h', 'coop'],
  'cooperative_groups/reduce.h': ['hip/cooperative_groups/hip_reduce.h', 'coop', 'partial', 'Only in recent HIP releases; on older ROCm use a shuffle loop or hipCUB WarpReduce.'],
  'cooperative_groups/scan.h': ['hip/cooperative_groups/hip_scan.h', 'coop', 'partial', 'Only in recent HIP releases.'],
  'helper_cuda.h': ['hip/hip_runtime_api.h', 'runtime', 'partial', 'NVIDIA sample helper. checkCudaErrors, findCudaDevice and friends are not in HIP; bring your own.'],
  'helper_math.h': ['hip/hip_vector_types.h', 'runtime', 'partial', 'NVIDIA sample helper. HIP vector types cover the operators but not every helper (lerp, clamp, reflect).'],
  'mma.h': ['rocwmma/rocwmma.hpp', 'wmma', 'partial', 'Port fragment code by hand.'],
  'nvToolsExt.h': ['roctracer/roctx.h', 'nvtx'],
  'nvtx3/nvToolsExt.h': ['roctracer/roctx.h', 'nvtx'],
  'texture_fetch_functions.h': ['hip/hip_runtime.h', 'runtime'],
  'cuda/pipeline': [null, 'runtime', 'unsupported', 'libcu++ pipelines have no HIP equivalent; use explicit loads into LDS and __syncthreads().'],
  'cuda/barrier': [null, 'runtime', 'unsupported', 'libcu++ barriers have no HIP equivalent.'],
  'cuda/atomic': [null, 'runtime', 'partial', 'Use std::atomic_ref or HIP atomics; libhipcxx is in preview.'],
}

const split = (s: string, sep: string) => new Map(s.split(' ').filter(Boolean).map(x => { const i = x.indexOf(sep); return [x.slice(0, i), x.slice(i + 1)] as [string, string] }))
const OFFICIAL = split(RENAMES, '=')
const OFFICIAL_EXPERIMENTAL = split(EXPERIMENTAL, '=')
const OFFICIAL_UNSUPPORTED = new Set(UNSUPPORTED.split(' '))
const OFFICIAL_DEVICE_UNSUPPORTED = new Set(DEVICE_UNSUPPORTED.split(' '))
const CUDA_DEPRECATED = split(DEPRECATED, ':')
const CUDA_REMOVED = split(REMOVED, ':')
const OFFICIAL_INCLUDES = split(INCLUDES, '=')

/** Library an identifier or header belongs to, from its prefix. */
export function libOf(id: string): Lib {
  if (/^cublasLt|^CUBLASLT/.test(id)) return 'cublaslt'
  if (/^cublas|^CUBLAS/.test(id)) return 'cublas'
  if (/^cufft|^CUFFT/.test(id)) return 'cufft'
  if (/^curand|^CURAND/.test(id)) return 'curand'
  if (/^cusparse|^CUSPARSE/.test(id)) return 'cusparse'
  if (/^cusolver|^CUSOLVER|^(?:gesvdj|syevj|csr\w+)Info/.test(id)) return 'cusolver'
  if (/^cutensor|^CUTENSOR/.test(id)) return 'cutensor'
  if (/^_?nvrtc|^NVRTC/.test(id)) return 'nvrtc'
  if (/^cu_?file|^IS_CUFILE/i.test(id)) return 'cufile'
  if (/^cudnn|^CUDNN/.test(id)) return 'cudnn'
  if (/^_*nv_|^__NV_/.test(id)) return 'fp'
  if (/^_*cub/i.test(id)) return 'cub'
  if (/^(?:make_)?cu(?:Float|Double)?Complex|^cuC[a-z]+$/.test(id)) return 'complex'
  if (/^cu[A-Z]|^CU[a-z_]|^CUDA_(?!R_|C_)|^IS_CUDA/.test(id)) return 'driver'
  return 'runtime'
}

const cudaAge = (id: string) => {
  const r = CUDA_REMOVED.get(id), d = CUDA_DEPRECATED.get(id)
  return r ? `Removed from CUDA in ${r}.` : d ? `Deprecated in CUDA since ${d}.` : undefined
}

/** Looks up a CUDA identifier. Returns null for identifiers that are not CUDA API names. */
export function lookup(id: string): Mapping | null {
  const row = T[id]
  if (row) return { hip: row[0], lib: row[1], status: row[2] ?? 'ok', note: row[3], via: 'table' }
  const hip = OFFICIAL.get(id)
  if (hip) return { hip, lib: libOf(id), status: 'ok', note: cudaAge(id), via: 'hipify' }
  const exp = OFFICIAL_EXPERIMENTAL.get(id)
  if (exp) return { hip: exp, lib: libOf(id), status: 'partial', note: 'Experimental in HIP: hipify only converts this with -experimental.', via: 'hipify' }
  if (OFFICIAL_UNSUPPORTED.has(id)) return { hip: null, lib: libOf(id), status: 'unsupported', note: ['hipify lists this as unsupported in HIP.', cudaAge(id)].filter(Boolean).join(' '), via: 'hipify' }
  if (OFFICIAL_DEVICE_UNSUPPORTED.has(id)) return { hip: null, lib: libOf(id), status: 'unsupported', note: 'Device function or type with no HIP implementation (hipify device table).', via: 'hipify' }
  if (/^cudnn[A-Z]\w*$|^CUDNN_\w+$/.test(id)) return { hip: null, lib: 'cudnn', status: 'unsupported', note: 'No mechanical MIOpen equivalent; port this call by hand.', via: 'family' }
  if (/^nccl[A-Z]\w*$/.test(id)) return { hip: id, lib: 'nccl', status: 'ok', via: 'family' }
  return null
}

/**
 * hipify always writes the FNUZ FP8 names. HIP compiles those out of device code on OCP targets
 * (amd_hip_fp8.h, HIP_FP8_TYPE_OCP / HIP_FP8_TYPE_FNUZ), where the plain names are the ones that exist.
 */
export function fp8Name(m: Mapping | null, fp8: 'none' | 'fnuz' | 'ocp' | undefined): Mapping | null {
  if (!m?.hip || !/^(?:__hip_fp8\w*_e[45]m[32]_fnuz|__HIP_E[45]M[32]_FNUZ)$/.test(m.hip)) return m
  if (fp8 === 'ocp') return { ...m, hip: m.hip.replace(/_fnuz$/i, ''), status: 'ok', note: 'OCP type: the same encoding as CUDA on this target. gfx942 (MI300) needs the _fnuz type instead.' }
  return { ...m, status: 'partial', note: m.note ?? 'hipify maps to the FNUZ type (gfx942, MI300): different bias and NaN encoding from CUDA. On gfx950 and RDNA4 use the OCP type.' }
}

// hipify still emits the flat pre-ROCm 6 header names for these; the packages install them here.
const ROCM_PATH: Record<string, string> = { 'hipblas.h': 'hipblas/hipblas.h', 'hipblaslt.h': 'hipblaslt/hipblaslt.h', 'hipsolver.h': 'hipsolver/hipsolver.h', 'hipsparse.h': 'hipsparse/hipsparse.h' }

export function lookupHeader(h: string): Mapping | null {
  const row = HEADERS[h]
  if (row) return { hip: row[0], lib: row[1], status: row[2] ?? 'ok', note: row[3], via: 'table' }
  const off = OFFICIAL_INCLUDES.get(h)
  if (off !== undefined) return { hip: ROCM_PATH[off] ?? (off || 'hip/hip_runtime.h'), lib: libOf(h), status: 'ok', via: 'hipify' }
  if (h.startsWith('thrust/')) return { hip: h, lib: 'thrust', status: 'ok', note: 'rocThrust keeps the thrust/ header paths.', via: 'family' }
  if (h.startsWith('cub/')) return { hip: 'hipcub/hipcub.hpp', lib: 'cub', status: 'ok', via: 'family' }
  if (/^cuda\//.test(h)) return { hip: null, lib: 'runtime', status: 'unsupported', note: 'libcu++ header; check libhipcxx or rewrite.', via: 'family' }
  return null
}

export const TABLE_SIZE = Object.keys(T).length + Object.keys(HEADERS).length + OFFICIAL.size + OFFICIAL_UNSUPPORTED.size
export const CURATED: Readonly<Record<string, readonly [string | null, Lib, Status?, string?]>> = T
