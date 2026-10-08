// D05 boundary test: 128-bit (vector) atomicAdd is sm_90+ only.
// Expected: compilation FAILS with -arch=sm_89 (Ada). The failure message is
// the evidence. See ../b128_sm89_build.log.
#include <cuda_runtime.h>

__global__ void k_vec_add(float4 *p, float4 v) { atomicAdd(p, v); }
