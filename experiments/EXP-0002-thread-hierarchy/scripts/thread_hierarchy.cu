// EXP-0002 - Thread hierarchy & scheduling semantics on sm_89 (Ada, RTX 4060 Laptop).
// Build & run: scripts/run.ps1
//
// Modes:
//   sched      record the global order in which blocks START and FINISH, plus the
//              SM each block landed on. Shows scheduling order is NOT blockIdx order
//              and is not stable across runs.
//   occupancy  occupancy is limited by threads AND registers AND shared memory;
//              `__launch_bounds__(maxTPB, minBlocksPerSM)` trades registers for
//              residency (and can spill to local memory).
//   barrier    divergent __syncthreads() for `compute-sanitizer --tool synccheck`.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t e = (x);                                                     \
    if (e != cudaSuccess) {                                                  \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,    \
             __LINE__);                                                      \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// sched: blocks report a global atomic "ticket" on entry and on exit.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned read_smid() {
  unsigned v;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
  return v;
}

__global__ void sched_kernel(int *ctr, int *start_tk, int *end_tk,
                             unsigned *smid) {
  int bid = blockIdx.x;
  if (threadIdx.x == 0) {
    start_tk[bid] = atomicAdd(ctr, 1);
    smid[bid] = read_smid();
  }
  // data-dependent, per-block variable work so finish order != start order
  long n = (long)((unsigned)bid * 2654435761u) % 97;
  volatile float acc = 0.f;
  for (long i = 0; i < (n + 1) * 100; ++i) acc += (float)(i & 7);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (acc == 12345.678f) start_tk[bid] = -42; // keep acc live
    end_tk[bid] = atomicAdd(ctr, 1);
  }
}

struct Ev {
  int ticket;
  int bid;
  int end;
};
static int ev_cmp(const void *a, const void *b) {
  int ta = ((const Ev *)a)->ticket, tb = ((const Ev *)b)->ticket;
  return (ta > tb) - (ta < tb);
}

static int run_sched() {
  const int G = 1024;
  int *dctr, *dstart, *dend;
  unsigned *dsmid;
  CUDA_CHECK(cudaMalloc(&dctr, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dstart, G * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dend, G * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dsmid, G * sizeof(unsigned)));

  int *start = (int *)malloc(G * sizeof(int));
  int *end = (int *)malloc(G * sizeof(int));
  unsigned *smid = (unsigned *)malloc(G * sizeof(unsigned));
  Ev *ev = (Ev *)malloc(2 * G * sizeof(Ev));

  char prev[4096] = {0};
  for (int run = 0; run < 3; ++run) {
    CUDA_CHECK(cudaMemset(dctr, 0, sizeof(int)));
    sched_kernel<<<G, 128>>>(dctr, dstart, dend, dsmid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(start, dstart, G * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(end, dend, G * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(smid, dsmid, G * sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int b = 0; b < G; ++b) {
      ev[b].ticket = start[b]; ev[b].bid = b; ev[b].end = 0;
      ev[G + b].ticket = end[b]; ev[G + b].bid = b; ev[G + b].end = 1;
    }
    qsort(ev, 2 * G, sizeof(Ev), ev_cmp);

    int *sorder = (int *)malloc(G * sizeof(int));
    int *eorder = (int *)malloc(G * sizeof(int));
    int si = 0, ei = 0;
    for (int i = 0; i < 2 * G; ++i) {
      if (ev[i].end) eorder[ei++] = ev[i].bid;
      else sorder[si++] = ev[i].bid;
    }
    bool s_asc = true, e_asc = true, same = true;
    for (int i = 0; i < G; ++i) {
      if (sorder[i] != i) s_asc = false;
      if (eorder[i] != i) e_asc = false;
      if (sorder[i] != eorder[i]) same = false;
    }
    printf("run %d: start==blockIdx? %s | finish==blockIdx? %s | finish==start? %s\n",
           run, s_asc ? "yes" : "NO", e_asc ? "yes" : "NO", same ? "yes" : "NO");
    printf("  start  order[0..31] :");
    for (int i = 0; i < 32; ++i) printf(" %d", sorder[i]);
    printf("\n  finish order[0..31] :");
    for (int i = 0; i < 32; ++i) printf(" %d", eorder[i]);
    printf("\n");
    char sig[4096];
    int p = 0;
    for (int i = 0; i < 64 && p < 4000; ++i)
      p += snprintf(sig + p, sizeof(sig) - p, "%d,", sorder[i]);
    if (run > 0)
      printf("  start-order[0..63] identical to previous run? %s\n",
             strcmp(sig, prev) == 0 ? "yes" : "NO");
    strncpy(prev, sig, sizeof(prev) - 1);
    prev[sizeof(prev) - 1] = 0;
    free(sorder);
    free(eorder);
  }
  cudaFree(dctr); cudaFree(dstart); cudaFree(dend); cudaFree(dsmid);
  free(start); free(end); free(smid); free(ev);
  return 0;
}

// ---------------------------------------------------------------------------
// occupancy: register / shared-memory pressure and __launch_bounds__.
// ---------------------------------------------------------------------------
__global__ void occ_plain(const float *in, float *out) {
  int i = threadIdx.x;
  float v = in[i & 1023];
  out[i] = v * 1.5f + 1.0f;
}

__global__ void occ_regs(const float *in, float *out) {
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = in[(threadIdx.x + i * 7) & 1023];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = fmaf(acc[i], 1.0001f, acc[(i + 1) & 63]);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 64; ++i) s += acc[i];
  out[threadIdx.x] = s;
}

// minBlocksPerMultiprocessor = 8 with 256 threads asks for 2048 threads/SM,
// which exceeds maxThreadsPerMultiProcessor (1536) -> ptxas ignores it.
__global__ __launch_bounds__(256, 8) void occ_lb(const float *in, float *out) {
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = in[(threadIdx.x + i * 7) & 1023];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = fmaf(acc[i], 1.0001f, acc[(i + 1) & 63]);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 64; ++i) s += acc[i];
  out[threadIdx.x] = s;
}

// Feasible request: 128 * 8 = 1024 threads/SM <= 1536 -> ptxas caps registers
// at 65536/(128*8) = 64 to honour 8 resident blocks (and may spill).
__global__ __launch_bounds__(128, 8) void occ_lb2(const float *in, float *out) {
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = in[(threadIdx.x + i * 7) & 1023];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = fmaf(acc[i], 1.0001f, acc[(i + 1) & 63]);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 64; ++i) s += acc[i];
  out[threadIdx.x] = s;
}

__global__ void occ_smem(const float *in, float *out) {
  __shared__ float sh[4096]; // 16 KB static shared memory
  int i = threadIdx.x;
  sh[i & 4095] = in[i & 1023];
  __syncthreads();
  out[i] = sh[(i + 1) & 4095] * 2.0f;
}

#define REPORT(kern, block)                                                   \
  do {                                                                        \
    cudaFuncAttributes a;                                                     \
    CUDA_CHECK(cudaFuncGetAttributes(&a, (const void *)kern));                \
    int nb = 0;                                                               \
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern,       \
                                                             block, 0));      \
    printf("%-10s block=%-4d regs=%-3d smem=%-6zu local=%-5zu maxTPB=%-4d -> " \
           "%d blocks/SM (%d threads/SM)\n",                                  \
           #kern, block, a.numRegs, a.sharedSizeBytes, a.localSizeBytes,      \
           a.maxThreadsPerBlock, nb, nb * block);                             \
  } while (0)

static int run_occupancy() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("device: %s  cc %d.%d  sm_%d%d\n", p.name, p.major, p.minor, p.major,
         p.minor);
  printf("multiProcessorCount          = %d\n", p.multiProcessorCount);
  printf("maxThreadsPerMultiProcessor  = %d\n", p.maxThreadsPerMultiProcessor);
  printf("regsPerMultiprocessor        = %d\n", p.regsPerMultiprocessor);
  printf("sharedMemPerMultiprocessor   = %zu\n", p.sharedMemPerMultiprocessor);
  printf("sharedMemPerBlock            = %zu\n", p.sharedMemPerBlock);
  printf("maxThreadsPerBlock           = %d\n", p.maxThreadsPerBlock);
  printf("warpSize                     = %d\n", p.warpSize);
  printf("---\n");
  const int N = 1024;
  float *din, *dout;
  CUDA_CHECK(cudaMalloc(&din, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, 32 * sizeof(float)));
  CUDA_CHECK(cudaMemset(din, 0, N * sizeof(float)));
  REPORT(occ_plain, 128);
  REPORT(occ_regs, 128);
  REPORT(occ_lb, 128);
  REPORT(occ_lb, 256);
  REPORT(occ_lb2, 128);
  REPORT(occ_smem, 128);
  // touch them so they are linked
  occ_plain<<<1, 32>>>(din, dout);
  occ_regs<<<1, 32>>>(din, dout);
  occ_lb<<<1, 32>>>(din, dout);
  occ_lb2<<<1, 32>>>(din, dout);
  occ_smem<<<1, 32>>>(din, dout);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(din);
  cudaFree(dout);
  return 0;
}

// ---------------------------------------------------------------------------
// barrier: a divergent __syncthreads() (only half the block ever arrives).
// ---------------------------------------------------------------------------
__global__ void barrier_deadlock(int *out) {
  if (threadIdx.x < 16) {
    while (true) __nanosleep(1000); // never reach the barrier
  }
  __syncthreads();
  out[threadIdx.x] = 1;
}

static int run_barrier() {
  int *dout;
  CUDA_CHECK(cudaMalloc(&dout, 32 * sizeof(int)));
  barrier_deadlock<<<1, 32>>>(dout);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); // hangs if not aborted by the tool
  printf("barrier kernel completed (no error surfaced)\n");
  cudaFree(dout);
  return 0;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "sched";
  if (!strcmp(mode, "sched")) return run_sched();
  if (!strcmp(mode, "occupancy")) return run_occupancy();
  if (!strcmp(mode, "barrier")) return run_barrier();
  printf("usage: %s {sched|occupancy|barrier}\n", argv[0]);
  return 2;
}
