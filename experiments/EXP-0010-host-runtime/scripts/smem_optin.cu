// EXP-0010 — dynamic shared memory opt-in boundary (launch-time, synchronous).
// default dynamic smem ceiling is sharedMemPerBlock (48 KiB here);
// larger needs cudaFuncSetAttribute(...MaxDynamicSharedMemorySize).
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__global__ void k(int *out) {
  extern __shared__ int s[];
  s[threadIdx.x] = threadIdx.x;
  out[threadIdx.x] = s[threadIdx.x];
}

int main() {
  cudaDeviceProp p; std::memset(&p, 0, sizeof(p));
  cudaGetDeviceProperties(&p, 0);
  printf("sharedMemPerBlock=%zu  optin=%zu\n",
         (size_t)p.sharedMemPerBlock, (size_t)p.sharedMemPerBlockOptin);

  int *d; cudaMalloc(&d, 256 * sizeof(int));
  const size_t big = 64 * 1024;   // 64 KiB > 48 KiB default

  k<<<1, 256, big>>>(d);
  printf("64 KiB BEFORE opt-in : %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaError_t a = cudaFuncSetAttribute(
      k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)big);
  printf("cudaFuncSetAttribute : %s\n", cudaGetErrorString(a));

  k<<<1, 256, big>>>(d);
  printf("64 KiB AFTER  opt-in : %s\n", cudaGetErrorString(cudaGetLastError()));
  printf("sync                 : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

  cudaFree(d);
  return 0;
}
