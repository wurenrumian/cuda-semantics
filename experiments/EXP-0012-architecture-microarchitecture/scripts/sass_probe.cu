// EXP-0012 — small kernels to compare SASS across -arch targets.
#include <cuda_runtime.h>

__global__ void sass_probe(const float *__restrict__ a,
                           const float *__restrict__ b, float *__restrict__ c,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = __ldg(a + i);
    float y = __ldg(b + i);
    c[i] = fmaf(x, 1.5f, y);
  }
}

__global__ void atom_probe(int *out) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  atomicAdd(&s, 1);
  __syncthreads();
  if (threadIdx.x == 0) *out = s;
}
