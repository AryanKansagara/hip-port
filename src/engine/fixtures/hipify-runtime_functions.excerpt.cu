// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#if defined(_WIN32)
  #include "windows.h"
  #include <GL/glew.h>
#endif

#include "cuda_gl_interop.h"
// CHECK: #include "hip/hip_runtime_api.h"
#include "cuda_profiler_api.h"

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

int main() {
  printf("12. CUDA Runtime API Functions synthetic test\n");

  size_t bytes = 0;
  // ... excerpt: sections of the upstream test are omitted here ...
  // CHECK: result = hipEventSynchronize(Event_t);
  result = cudaEventSynchronize(Event_t);

  // CHECK: hipFuncAttributes FuncAttributes;
  cudaFuncAttributes FuncAttributes;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
  // HIP: hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func);
  // CHECK: result = hipFuncGetAttributes(&FuncAttributes, reinterpret_cast<const void*>(func));
  result = cudaFuncGetAttributes(&FuncAttributes, func);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig);
  // HIP: hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config);
  // CHECK: result = hipFuncSetCacheConfig(reinterpret_cast<const void*>(func), FuncCache);
  result = cudaFuncSetCacheConfig(func, FuncCache);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config);
  // HIP: hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config);
  // CHECK: result = hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(func), SharedMemConfig);
  result = cudaFuncSetSharedMemConfig(func, SharedMemConfig);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
  // HIP: hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes __dparm(0), hipStream_t stream __dparm(0));
  // CHECK: result = hipLaunchKernel(reinterpret_cast<const void*>(func), gridDim, blockDim, &image, bytes, stream);
  result = cudaLaunchKernel(func, gridDim, blockDim, &image, bytes, stream);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
  // HIP: hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk);
  // CHECK: result = hipOccupancyMaxActiveBlocksPerMultiprocessor(&intVal, func, device, bytes);
  result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&intVal, func, device, bytes);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
  // HIP: hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags __dparm(hipOccupancyDefault));
  // CHECK: result = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&intVal, func, intVal, bytes, flags);
  result = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&intVal, func, intVal, bytes, flags);

  // CUDA: template<class T> static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0);
  // HIP: template <typename T> static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0);
  // CHECK: result = hipOccupancyMaxPotentialBlockSize(&intVal, &device, func, bytes, deviceId);
  result = cudaOccupancyMaxPotentialBlockSize(&intVal, &device, func, bytes, deviceId);

  // CUDA: template<class T> static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0, unsigned int flags = 0);
  // HIP: template <typename T> static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize, T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0, unsigned int flags = 0);
  // CHECK: result = hipOccupancyMaxPotentialBlockSizeWithFlags(&intVal, &device, func, bytes, deviceId, flags);
  result = cudaOccupancyMaxPotentialBlockSizeWithFlags(&intVal, &device, func, bytes, deviceId, flags);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr);
  // HIP: hipError_t hipFree(void* ptr);
  // CHECK: result = hipFree(deviceptr);
  result = cudaFree(deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array);
  // HIP: hipError_t hipFreeArray(hipArray* array);
  // CHECK: result = hipFreeArray(Array_t);
  result = cudaFreeArray(Array_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr);
  // HIP: hipError_t hipHostFree(void* ptr);
  // CHECK: result = hipHostFree(deviceptr);
  result = cudaFreeHost(deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
  // HIP: hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);
  // CHECK: result = hipFreeMipmappedArray(MipmappedArray_t);
  result = cudaFreeMipmappedArray(MipmappedArray_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
  // HIP: hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray, hipMipmappedArray_const_t mipmappedArray, unsigned int level);
  // CHECK: result = hipGetMipmappedArrayLevel(&Array_t, MipmappedArray_const_t, flags);
  result = cudaGetMipmappedArrayLevel(&Array_t, MipmappedArray_const_t, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol);
  // HIP: hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol);
  // CHECK: result = hipGetSymbolAddress(&deviceptr, HIP_SYMBOL(image));
  result = cudaGetSymbolAddress(&deviceptr, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol);
  // HIP: hipError_t hipGetSymbolSize(size_t* size, const void* symbol);
  // CHECK: result = hipGetSymbolSize(&bytes, HIP_SYMBOL(image));
  result = cudaGetSymbolSize(&bytes, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
  // HIP: DEPRECATED("use hipHostMalloc instead") hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
  // CHECK: result = hipHostAlloc(&deviceptr, bytes, flags);
  result = cudaHostAlloc(&deviceptr, bytes, flags);

  result = cudaMemcpy3DAsync(&Memcpy3DParms, stream);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyAsync(deviceptr, deviceptr_2, bytes, MemcpyKind, stream);
  result = cudaMemcpyAsync(deviceptr, deviceptr_2, bytes, MemcpyKind, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
  // HIP: hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset __dparm(0), hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost));
  // CHECK: result = hipMemcpyFromSymbol(deviceptr, HIP_SYMBOL(image), bytes, wOffset, MemcpyKind);
  result = cudaMemcpyFromSymbol(deviceptr, image, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyFromSymbolAsync(deviceptr, HIP_SYMBOL(image), bytes, wOffset, MemcpyKind, stream);
  result = cudaMemcpyFromSymbolAsync(deviceptr, image, bytes, wOffset, MemcpyKind, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
  // HIP: hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes);
  // CHECK: result = hipMemcpyPeer(deviceptr, deviceId, deviceptr_2, device, bytes);
  result = cudaMemcpyPeer(deviceptr, deviceId, deviceptr_2, device, bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice, size_t sizeBytes, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyPeerAsync(deviceptr, deviceId, deviceptr_2, device, bytes, stream);
  result = cudaMemcpyPeerAsync(deviceptr, deviceId, deviceptr_2, device, bytes, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
  // HIP: hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset __dparm(0), hipMemcpyKind kind __dparm(hipMemcpyHostToDevice));
  // CHECK: result = hipMemcpyToSymbol(HIP_SYMBOL(image), deviceptr, bytes, wOffset, MemcpyKind);
  result = cudaMemcpyToSymbol(image, deviceptr, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyToSymbolAsync(HIP_SYMBOL(image), deviceptr, bytes, wOffset, MemcpyKind, stream);
  result = cudaMemcpyToSymbolAsync(image, deviceptr, bytes, wOffset, MemcpyKind, stream);

  return 0;
}
