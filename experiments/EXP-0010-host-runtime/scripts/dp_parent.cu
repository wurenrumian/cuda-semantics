// EXP-0010 — dynamic parallelism (device-side launch) boundary check.
// A parent __global__ launches a child __global__.  This requires separate
// compilation of relocatable device code: -rdc=true and the device runtime
// (-lcudadevrt).  Without them the build fails — that failure IS the point.
//
// Build (works): vcvars64.bat ; nvcc -O3 -arch=sm_89 -rdc=true -lcudadevrt -o dp_parent.exe dp_parent.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void child(int *out) { out[0] = 42; }

__global__ void parent(int *out) {
  // Device-side launch. CUDA 12 defaults to "CDP2": the child goes to the
  // parent's tail (tail-launch semantics), so the host sync below still waits
  // for it. NOTE: device-side cudaDeviceSynchronize() is a CDP1 API that CUDA
  // 12.8 no longer even declares with -rdc (see cuda_device_runtime_api.h).
  child<<<1, 1>>>(out);
}

int main() {
  int *d = nullptr;
  cudaMalloc(&d, sizeof(int));
  cudaMemset(d, 0, sizeof(int));
  parent<<<1, 1>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  printf("parent sync: %s\n", cudaGetErrorString(e));
  int h = -1;
  cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  printf("child wrote %d (expect 42)\n", h);
  cudaFree(d);
  return 0;
}
