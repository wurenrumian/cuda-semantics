// Compile-only probe: does ptxas accept a named barrier whose count is not a
// multiple of the warp size?  PTX ISA requires count % warpSize == 0.
#include <cuda_runtime.h>

__global__ void k_bad(int *out) {
  int t = threadIdx.x;
  out[t] = t;
  asm volatile("bar.sync 1, 16;" : : : "memory");   // 16 is not a multiple of 32
}
