// EXP-0006 / domain 06 — synchronization primitives.
// Target: sm_89 (Ada, RTX 4060 Laptop), CUDA 12.8.  Build & run: scripts/run.ps1
//
// Modes (argv[1]):
//   smem-order-unsync          — shared-memory neighbor read, NO barrier (data race)
//   smem-order-sync            — same read behind __syncthreads()          (clean)
//   divergent-syncthreads      — __syncthreads() in a divergent branch; missing
//                                threads EXIT first -> completes (still UB!)
//   divergent-syncthreads-alive— same, but the missing threads SPIN forever-> wedges
//   divergent-syncwarp         — __syncwarp() (default mask) in a divergent branch
//   syncwarp-partial-alive     — half-warp __syncwarp(), other half SPINS -> wedges
//   named-barrier              — two independent named barriers partition the block
//
// The "*alive" modes wedge ON PURPOSE: a sibling thread spins on a flag that is
// never set, so the barrier it never reaches can never be satisfied. This is a
// deliberate non-terminating demo of a divergent barrier deadlock, not a bug.
// A host-side watchdog polls cudaStreamQuery (hard deadline ~0.5 s; measured
// wall time is printed) and reports the wedge, then _exit(0) so a deadlocked
// barrier is observable without hanging the shell.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
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
// Positive: __syncthreads() orders shared-memory writes/reads across the block.
// ---------------------------------------------------------------------------
__global__ void k_smem_order_unsync(int *out) {
  __shared__ int sh[256];
  int t = threadIdx.x;
  sh[t] = t * 2;
  out[t] = sh[(t + 1) & 255];    // races with the neighbour's write
}

__global__ void k_smem_order_sync(int *out) {
  __shared__ int sh[256];
  int t = threadIdx.x;
  sh[t] = t * 2;
  __syncthreads();
  out[t] = sh[(t + 1) & 255];
}

// ---------------------------------------------------------------------------
// Divergent __syncthreads(): threads with t >= blockDim/2 take the else path.
// Variant A: they EXIT -> the barrier is satisfied by the remaining live threads
//            (compiler emits @P0 EXIT).  UB by the standard, but it "works".
// Variant B: they SPIN on a global flag that never changes -> the barrier can
//            never be satisfied -> the block wedges.
// ---------------------------------------------------------------------------
__global__ void k_divergent_syncthreads_exit(int *out) {
  int t = threadIdx.x;
  if (t < blockDim.x / 2) {
    out[t] = 1;
    __syncthreads();
  } else {
    out[t] = 2;
  }
}

__global__ void k_divergent_syncthreads_alive(int *out, volatile int *spin) {
  int t = threadIdx.x;
  if (t < blockDim.x / 2) {
    out[t] = 1;
    __syncthreads();             // waits for the whole block
  } else {
    while (*spin == 0) { }       // sibling stays alive, never reaches the barrier
    out[t] = 2;
  }
}

// ---------------------------------------------------------------------------
// Divergent __syncwarp(): default mask 0xffffffff, only the low half arrives.
// ---------------------------------------------------------------------------
__global__ void k_divergent_syncwarp(int *out) {
  int lane = threadIdx.x & 31;
  if (lane < 16) {
    out[threadIdx.x] = 1;
    __syncwarp();
  } else {
    out[threadIdx.x] = 2;
  }
}

__global__ void k_syncwarp_partial_alive(int *out, volatile int *spin) {
  int lane = threadIdx.x & 31;
  if (lane < 16) {
    out[threadIdx.x] = 1;
    __syncwarp();                // names all 32 lanes ...
  } else {
    while (*spin == 0) { }       // ... but these stay alive, never arrive
    out[threadIdx.x] = 2;
  }
}

// ---------------------------------------------------------------------------
// Named barriers: split a 256-thread block into two halves that synchronize
// independently via bar.sync 1,128 / bar.sync 2,128 (inline PTX).
// ---------------------------------------------------------------------------
__global__ void k_named_barrier(int *out) {
  __shared__ int sh[256];
  int t = threadIdx.x;
  sh[t] = t * 3;
  if (t < 128) {
    asm volatile("bar.sync 1, 128;" : : : "memory");
    out[t] = sh[(t + 1) & 127];
  } else {
    asm volatile("bar.sync 2, 128;" : : : "memory");
    out[t] = sh[128 + ((t - 128 + 1) & 127)];
  }
}

// ---------------------------------------------------------------------------
// Host-side watchdog: launch, poll cudaStreamQuery with a hard ~0.5 s deadline
// (50 polls x 10 ms; the old 20M-iteration busy loop measured ~0.25 s for all
// 50 polls, far below the "~5 s" it assumed), report outcome with the measured
// elapsed time.
// ---------------------------------------------------------------------------
static void run_with_watchdog(void (*k)(int *, volatile int *), int n,
                              const char *label) {
  int *d;
  volatile int *spin;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&spin, sizeof(int)));
  CUDA_CHECK(cudaMemset((void *)spin, 0, sizeof(int)));   // never changes
  k<<<1, n>>>(d, spin);
  CUDA_CHECK(cudaGetLastError());
  cudaError_t e = cudaErrorNotReady;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 50; ++i) {                 // 50 polls x 10 ms ~= 0.5 s deadline
    e = cudaStreamQuery(0);
    if (e != cudaErrorNotReady) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  double held =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (e == cudaSuccess) {
    int h0 = -1;
    cudaMemcpy(&h0, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%s: FINISHED (block completed; out[0]=%d)\n", label, h0);
  } else if (e == cudaErrorNotReady) {
    printf("%s: WEDGED — kernel still unfinished after %.2f s (barrier never satisfied)\n",
           label, held);
    fflush(stdout);
    _exit(0);                                    // abandon the stuck context
  } else {
    printf("%s: kernel aborted: %s (driver/TDR intervened)\n", label,
           cudaGetErrorString(e));
    cudaGetLastError();
  }
}

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "help";

  if (mode == "smem-order-unsync" || mode == "smem-order-sync") {
    const int n = 256;
    int *d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    if (mode == "smem-order-unsync")
      k_smem_order_unsync<<<1, n>>>(d);
    else
      k_smem_order_sync<<<1, n>>>(d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int h[256];
    CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i)
      if (h[i] != ((i + 1) & 255) * 2) bad++;
    printf("%s: %d/%d wrong values (a race may still look correct)\n",
           mode.c_str(), bad, n);
    cudaFree(d);

  } else if (mode == "divergent-syncthreads") {
    const int n = 256;
    int *d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    k_divergent_syncthreads_exit<<<1, n>>>(d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("divergent-syncthreads(exit): FINISHED — missing threads exited, so "
           "BAR.SYNC 0 was still satisfied (still UB by the standard)\n");
    cudaFree(d);

  } else if (mode == "divergent-syncthreads-alive") {
    run_with_watchdog(k_divergent_syncthreads_alive, 256,
                      "divergent-syncthreads(alive)");

  } else if (mode == "divergent-syncwarp") {
    const int n = 64;
    int *d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    k_divergent_syncwarp<<<1, 64>>>(d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("divergent-syncwarp(exit): FINISHED — WARPSYNC named lanes that had "
           "already exited (still UB)\n");
    cudaFree(d);

  } else if (mode == "syncwarp-partial-alive") {
    run_with_watchdog(k_syncwarp_partial_alive, 64, "syncwarp-partial(alive)");

  } else if (mode == "named-barrier") {
    const int n = 256;
    int *d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    k_named_barrier<<<1, n>>>(d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int h[256];
    CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 128; ++i)
      if (h[i] != ((i + 1) & 127) * 3) bad++;
    printf("named-barrier: group0 correct %d/128\n", 128 - bad);
    for (int i = 0; i < 4; ++i) printf("  out[%d]=%d\n", i, h[i]);
    cudaFree(d);

  } else {
    printf("usage: %s {smem-order-unsync|smem-order-sync|divergent-syncthreads|"
           "divergent-syncthreads-alive|divergent-syncwarp|syncwarp-partial-alive|"
           "named-barrier}\n", argv[0]);
    return 2;
  }
  return 0;
}
