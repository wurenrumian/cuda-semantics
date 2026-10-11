// EXP-0012 — architecture / microarchitecture probe.
// Target: sm_89 (Ada, RTX 4060 Laptop), CUDA 12.8.
// Build & run: scripts/run.ps1
//
// Demonstrates, on the local machine:
//   (1) the architecture facts you can RELY on are exactly the queryable ones
//       (cudaDeviceProp / cudaFuncGetAttributes) — print them;
//   (2) block->SM placement and the SM count (via %smid);
//   (3) occupancy is shaped by maxThreadsPerMultiProcessor (1536 on Ada, NOT 2048);
//   (4) >48KB shared memory needs dynamic smem + an explicit opt-in.
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x)                                                              \
  do {                                                                        \
    cudaError_t e_ = (x);                                                     \
    if (e_ != cudaSuccess) {                                                  \
      printf("[CUDA ERROR] %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__,   \
             __LINE__);                                                       \
      fflush(stdout);                                                         \
    }                                                                         \
  } while (0)

__global__ void k_tiny(unsigned *out) { out[threadIdx.x] = threadIdx.x; }

__global__ void k_smid(unsigned *smid, unsigned *wid) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned s, w;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(w));
  smid[i] = s;
  wid[i] = w;
}

__global__ void k_dyn(unsigned *out, int n) {
  extern __shared__ unsigned char sh[];
  if (threadIdx.x == 0) out[0] = sh[0] + sh[n - 1];
}

int main() {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));

  printf("==== cudaDeviceProp (queryable architecture facts) ====\n");
  printf("name                          : %s\n", p.name);
  printf("compute capability            : %d.%d\n", p.major, p.minor);
  printf("multiProcessorCount           : %d\n", p.multiProcessorCount);
  printf("warpSize                      : %d\n", p.warpSize);
  printf("maxThreadsPerBlock            : %d\n", p.maxThreadsPerBlock);
  printf("maxThreadsPerMultiProcessor   : %d\n", p.maxThreadsPerMultiProcessor);
  printf("maxBlocksPerMultiProcessor    : %d\n", p.maxBlocksPerMultiProcessor);
  printf("regsPerBlock                  : %d\n", p.regsPerBlock);
  printf("regsPerMultiprocessor         : %d\n", p.regsPerMultiprocessor);
  printf("sharedMemPerBlock             : %zu\n", p.sharedMemPerBlock);
  printf("sharedMemPerMultiprocessor    : %zu\n", p.sharedMemPerMultiprocessor);
  printf("sharedMemPerBlockOptin        : %zu\n", p.sharedMemPerBlockOptin);
  printf("reservedSharedMemPerBlock     : %zu\n", p.reservedSharedMemPerBlock);
  printf("l2CacheSize                   : %d\n", p.l2CacheSize);
  printf("persistingL2CacheMaxSize      : %d\n", p.persistingL2CacheMaxSize);
  printf("memoryBusWidth                : %d\n", p.memoryBusWidth);
  printf("memoryClockRate (kHz)         : %d\n", p.memoryClockRate);
  printf("clockRate (kHz)               : %d\n", p.clockRate);
  printf("concurrentKernels             : %d\n", p.concurrentKernels);
  printf("unifiedAddressing             : %d\n", p.unifiedAddressing);

  printf("\n==== cudaFuncGetAttributes(k_tiny) ====\n");
  cudaFuncAttributes a;
  CHECK(cudaFuncGetAttributes(&a, k_tiny));
  // NOTE: cudaFuncAttributes::maxDynamicSharedSizeBytes is `int` (not size_t),
  // so it must be printed with %d; the other two byte counts are size_t.
  printf("numRegs=%d sharedSizeBytes=%zu maxThreadsPerBlock=%d "
         "maxDynamicSharedSizeBytes=%d localSizeBytes=%zu\n",
         a.numRegs, a.sharedSizeBytes, a.maxThreadsPerBlock,
         a.maxDynamicSharedSizeBytes, a.localSizeBytes);

  printf("\n==== occupancy vs block size (k_tiny, 0 dynamic smem) ====\n");
  int bs[] = {128, 256, 384, 512, 640, 768, 1024};
  for (int i = 0; i < (int)(sizeof(bs) / sizeof(bs[0])); ++i) {
    int blocks = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void *)k_tiny,
                                                        bs[i], 0));
    printf("block=%4d -> %d block/SM = %4d threads/SM (%.1f%% of %d)\n", bs[i],
           blocks, blocks * bs[i],
           100.0 * blocks * bs[i] / p.maxThreadsPerMultiProcessor,
           p.maxThreadsPerMultiProcessor);
  }

  printf("\n==== block -> SM placement via %%smid ====\n");
  int per_sm = 8;
  int nblk = p.multiProcessorCount * per_sm;
  int bsz = 128;
  int total = nblk * bsz;
  unsigned *d_smid = nullptr, *d_wid = nullptr;
  CHECK(cudaMalloc(&d_smid, total * sizeof(unsigned)));
  CHECK(cudaMalloc(&d_wid, total * sizeof(unsigned)));
  k_smid<<<nblk, bsz>>>(d_smid, d_wid);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  unsigned *h_smid = new unsigned[total];
  CHECK(cudaMemcpy(h_smid, d_smid, total * sizeof(unsigned),
                   cudaMemcpyDeviceToHost));
  int seen[1024] = {0};
  int distinct = 0;
  for (int i = 0; i < total; ++i) {
    if (h_smid[i] < 1024 && seen[h_smid[i]]++ == 0) ++distinct;
  }
  printf("launched %d blocks x %d threads on a %d-SM device\n", nblk, bsz,
         p.multiProcessorCount);
  printf("distinct SMs observed       : %d\n", distinct);
  int lo = 1 << 30, hi = -1;
  for (int i = 0; i < total; ++i) {
    if ((int)h_smid[i] < lo) lo = h_smid[i];
    if ((int)h_smid[i] > hi) hi = h_smid[i];
  }
  printf("smid range observed          : [%d, %d]\n", lo, hi);

  printf("\n==== dynamic shared memory opt-in ====\n");
  size_t optin = p.sharedMemPerBlockOptin;
  unsigned *d_out = nullptr;
  CHECK(cudaMalloc(&d_out, sizeof(unsigned)));

  // (a) request `optin` dynamic bytes WITHOUT opt-in
  k_dyn<<<1, 128, optin>>>(d_out, (int)optin);
  cudaError_t e1 = cudaGetLastError();
  printf("dynamic smem=%zu without opt-in : %s\n", optin,
         cudaGetErrorString(e1));
  cudaGetLastError(); // clear

  // (b) opt in explicitly, then retry
  cudaError_t e2 = cudaFuncSetAttribute(
      k_dyn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)optin);
  printf("cudaFuncSetAttribute(MaxDynamicSharedMemorySize=%zu) : %s\n", optin,
         cudaGetErrorString(e2));
  k_dyn<<<1, 128, optin>>>(d_out, (int)optin);
  cudaError_t e3 = cudaGetLastError();
  printf("dynamic smem=%zu WITH opt-in     : %s\n", optin,
         cudaGetErrorString(e3));
  CHECK(cudaDeviceSynchronize());

  delete[] h_smid;
  cudaFree(d_smid);
  cudaFree(d_wid);
  cudaFree(d_out);
  return 0;
}
