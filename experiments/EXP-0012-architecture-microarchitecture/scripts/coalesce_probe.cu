// EXP-0012 — 32B sector granularity: coalesced vs strided global loads.
// Same number of elements read per launch; only the per-lane address pattern
// differs. A warp that reads 32 contiguous floats touches 4 x 32B sectors;
// a warp whose lanes are 128B apart touches 32 x 32B sectors (8x).
//
// Ideal observation is the L1TEX sector/request counters via ncu:
//   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
//   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
// On this host ncu counters are blocked (ERR_NVGPUCTRPERM); we fall back to a
// wall-clock timing observation, which is qualitative only.
//
// Usage: coalesce_probe <coalesced|strided> [iters]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Contiguous: lane L reads in[base+L] -> warp touches 128B = 4 sectors.
__global__ void ld_coalesced(const float *__restrict__ in,
                             float *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

// Strided: warp w reads in[w*1024 + lane*32] -> each lane in its own sector.
// `in` is 32x larger so every access stays in bounds; exactly n elements are
// read, same as the coalesced kernel.
__global__ void ld_strided(const float *__restrict__ in,
                           float *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  int warp = i >> 5;
  int j = warp * 1024 + lane * 32;
  if (i < n) out[i] = in[j];
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "coalesced";
  int iters = argc > 2 ? atoi(argv[2]) : 1;
  const int n = 1 << 20; // 1M elements read per launch
  float *in, *out;
  cudaMalloc(&in, (size_t)32 * n * sizeof(float)); // 128 MB backing
  cudaMalloc(&out, (size_t)n * sizeof(float));
  cudaMemset(in, 1, (size_t)32 * n * sizeof(float));

  const int block = 256;
  const int grid = n / block;

  // warmup
  if (mode[0] == 'c')
    ld_coalesced<<<grid, block>>>(in, out, n);
  else
    ld_strided<<<grid, block>>>(in, out, n);
  cudaDeviceSynchronize();

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int k = 0; k < iters; ++k) {
    if (mode[0] == 'c')
      ld_coalesced<<<grid, block>>>(in, out, n);
    else
      ld_strided<<<grid, block>>>(in, out, n);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  printf("%s: %d iters, %.3f ms total, %.4f ms/iter\n", mode, iters, ms,
         ms / iters);

  cudaFree(in);
  cudaFree(out);
  return 0;
}
