// D05 diagnostic: minimal CUDA app to test compute-sanitizer injection.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int *o) { o[0] = 1; }
int main() {
  int *d; cudaMalloc(&d, sizeof(int));
  k<<<1, 1>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  int h; cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  printf("device=%d err=%s val=%d\n", h, cudaGetErrorString(e), h);
  cudaFree(d);
  return 0;
}
