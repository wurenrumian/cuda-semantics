// EXP-0012 — ISA feature availability is gated by the -arch target.
// cp.async requires sm_80+ (PTX ISA: ".target sm_80 or higher").
// Expected: compiles for sm_89, FAILS ptxas for sm_75.
#include <cuda_runtime.h>

__global__ void cp_probe(const unsigned *__restrict__ src,
                         unsigned *__restrict__ dst) {
  extern __shared__ unsigned char sh[];
  unsigned saddr = (unsigned)__cvta_generic_to_shared(sh);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(saddr),
               "l"(src));
  asm volatile("cp.async.commit_group;\n");
  asm volatile("cp.async.wait_group 0;\n");
  __syncthreads();
  if (threadIdx.x == 0) dst[0] = ((unsigned *)sh)[0];
}
