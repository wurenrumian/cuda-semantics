// D01 — SIMT execution model probes. Target: sm_89 (Ada) / CUDA 12.8.
// Build & run:  pwsh -File ..\run.ps1
//
// Modes:
//   mask            — dump __activemask() at several program points (converged vs divergent)
//   race-unsync     — warp-synchronous neighbour read WITHOUT __syncwarp (UB on Volta+)
//   race-sync       — same read with __syncwarp()
//   shfl-bad        — uses __activemask() as a __shfl_sync mask inside divergence (trap)
//   shfl-good       — convergent __shfl_sync(0xffffffff,...) doing the logical swap
//   intrinsics      — __ballot_sync / __reduce_add_sync / __match_any_sync
//   sass            — no-op tunable kernel, compiled for cuobjdump -sass inspection
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s @line %d\n", cudaGetErrorString(e), __LINE__);     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const char *hex(unsigned x) {
  static char b[16][12];
  static int k = 0;
  char *p = b[k = (k + 1) & 15];
  sprintf(p, "0x%08x", x);
  return p;
}

// ---------------------------------------------------------------------------
// __activemask() probe.  Prints the mask observed at:
//   c0  converged, before any branch
//   mt  lanes 0..15, inside the "then" arm
//   me  lanes 16..31, inside the "else" arm
//   c1  right after the if/else join
//   ml  read by the lanes that still iterate a divergent while-loop
//   c2  after the loop
// ---------------------------------------------------------------------------
__global__ void k_mask(unsigned *out) {
  int lane = threadIdx.x & 31;
  unsigned c0 = __activemask();
  unsigned mt = 0, me = 0;
  if (lane < 16) {
    mt = __activemask();
  } else {
    me = __activemask();
  }
  unsigned c1 = __activemask();
  unsigned ml = 0;
  int i = 0, lim = lane & 3; // 0..3 => lanes diverge in trip count
  while (i < lim) {
    ml = __activemask();
    ++i;
  }
  unsigned c2 = __activemask();
  out[0 * 32 + lane] = c0;
  out[1 * 32 + lane] = mt;
  out[2 * 32 + lane] = me;
  out[3 * 32 + lane] = c1;
  out[4 * 32 + lane] = ml;
  out[5 * 32 + lane] = c2;
}

// ---------------------------------------------------------------------------
// Warp-synchronous neighbour read.  Pre-Volta code relied on the warp being in
// lockstep: "I write sh[lane], by the time I read sh[lane^1] my neighbour has
// written too."  Not a guarantee on Volta+.
// ---------------------------------------------------------------------------
__global__ void k_neighbor_unsync(int *out) {
  __shared__ int sh[32];
  int lane = threadIdx.x & 31;
  sh[lane] = lane * 10;
  int n = sh[lane ^ 1]; // racy read of the neighbour's store
  out[lane] = n;
}

__global__ void k_neighbor_sync(int *out) {
  __shared__ int sh[32];
  int lane = threadIdx.x & 31;
  sh[lane] = lane * 10;
  __syncwarp(); // happens-before among the named lanes
  int n = sh[lane ^ 1];
  out[lane] = n;
}

// ---------------------------------------------------------------------------
// Warp reductions three ways.
// ---------------------------------------------------------------------------
// (a) register shuffle: reads the source lane's register directly -> correct.
__global__ void k_reduce_shfl(int *out) {
  int lane = threadIdx.x & 31;
  int v = threadIdx.x;
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  out[lane] = v; // lane 0 holds 0+..+31 = 496
}

// (b) shared memory with ONE __syncwarp() before the loop.  A barrier orders
// the writes before the loop, but each step reads values other lanes are still
// updating -> the step barrier is missing.  Classic warp-synchronous trap.
// The guard keeps lanes lane>=off from reading red[lane+off] past red[31].
__global__ void k_reduce_nosync(int *out) {
  __shared__ int red[32];
  int lane = threadIdx.x & 31;
  red[lane] = threadIdx.x;
  __syncwarp();
  for (int off = 16; off > 0; off >>= 1)
    if (lane < off)
      red[lane] += red[lane + off];
  out[lane] = red[lane];
}

// (c) same, with a __syncwarp() inside every step.  Only lane 0 is defined to
// hold the full sum; lanes >= off do not update in the final step.
__global__ void k_reduce_sync(int *out) {
  __shared__ int red[32];
  int lane = threadIdx.x & 31;
  red[lane] = threadIdx.x;
  for (int off = 16; off > 0; off >>= 1) {
    __syncwarp();
    if (lane < off)
      red[lane] += red[lane + off];
  }
  __syncwarp();
  out[lane] = red[lane];
}

// ---------------------------------------------------------------------------
// Independent Thread Scheduling: lane i does (i+1)*iters dependent ops.  If the
// warp ran in lockstep every lane would report the same duration.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long rdclock() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}

__global__ void k_timing(unsigned long long *out, long iters) {
  int lane = threadIdx.x & 31;
  unsigned long long t0 = rdclock();
  unsigned long long t1 = t0;
  if (lane > 0) { // divergent arm; stamp the clock before the join point
    volatile unsigned s = (unsigned)(lane + 1);
    long n = (long)lane * iters;
    for (long i = 0; i < n; ++i)
      s = s * 1103515245u + 12345u; // serial LCG, not closed-form
    t1 = rdclock();
    if (s == 0xdeadbeefu) t1 = 0; // keep the loop live
  }
  out[threadIdx.x] = t1 - t0;
}

// ---------------------------------------------------------------------------
// Trap: "I'll just re-derive the mask with __activemask()".
// Even lanes enter the branch and shuffle from lane+1 (an odd lane).  But
// __activemask() only names the lanes converged at THIS instruction, so the
// partner may not be in the mask -> the shuffle is undefined.
// ---------------------------------------------------------------------------
__global__ void k_shfl_bad(int *out) {
  int lane = threadIdx.x & 31;
  int v = lane * 100;
  out[lane] = -1;
  out[32 + lane] = 0;
  if ((lane & 1) == 0) {
    unsigned m = __activemask();          // looks like "the warp", is not
    int x = __shfl_sync(m, v, lane + 1);  // partner odd lane may be absent
    out[lane] = x;
    out[32 + lane] = (int)m;
  }
}

// Correct shape: stay converged, pass the full mask, compute indices.
__global__ void k_shfl_good(int *out) {
  int lane = threadIdx.x & 31;
  int v = lane * 100;
  int src = (lane & 1) ? (lane - 1) : (lane + 1);
  int x = __shfl_sync(0xffffffffu, v, src);
  out[lane] = x;
}

// ---------------------------------------------------------------------------
// A few warp-level primitives on full warp.
// ---------------------------------------------------------------------------
__global__ void k_intrinsics(int *out, int *extra) {
  int lane = threadIdx.x & 31;
  __shared__ int red[32];
  red[lane] = threadIdx.x;
  for (int off = 16; off > 0; off >>= 1) {
    __syncwarp();
    if (lane < off)
      red[lane] += red[lane + off];
  }
  __syncwarp();
  out[lane] = red[lane]; // only lane 0 ends with 0+..+31 = 496

  extra[0] = (int)__ballot_sync(0xffffffffu, lane & 1);   // 0xaaaaaaaa
  extra[1] = __reduce_add_sync(0xffffffffu, threadIdx.x); // 496
  unsigned same = __match_any_sync(0xffffffffu, lane >> 4); // groups of 16
  extra[2 + lane] = (int)same;
}

// ---------------------------------------------------------------------------
// Tunable no-op kernel so cuobjdump shows BSSY/BSYNC/BRA/WARPSYNC.
// ---------------------------------------------------------------------------
__global__ void k_sass(int *out, int flag) {
  int lane = threadIdx.x & 31;
  __shared__ int sh[32];
  sh[lane] = lane;
  __syncwarp();
  int acc = 0;
  if (flag) {
    for (int i = 0; i < lane; ++i)
      if (i & 1)
        acc += sh[i];
  } else {
    acc = sh[31 - lane];
  }
  out[lane] = acc;
}

// ---------------------------------------------------------------------------
static void dumpMask() {
  unsigned *d;
  CK(cudaMalloc(&d, 6 * 32 * sizeof(unsigned)));
  k_mask<<<1, 32>>>(d);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  unsigned h[6 * 32];
  CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  const char *names[6] = {"c0 converged      (expect 0xffffffff)",
                          "mt then  lane<16  (expect 0x0000ffff)",
                          "me else  lane>=16 (expect 0xffff0000)",
                          "c1 after join     (expect 0xffffffff)",
                          "ml divergent while (varies)",
                          "c2 after loop     (expect 0xffffffff)"};
  for (int r = 0; r < 6; ++r) {
    printf("%-38s :", names[r]);
    for (int l = 0; l < 32; ++l) {
      unsigned m = h[r * 32 + l];
      // show placeholder if this lane never executed the read
      printf(" %s", (r == 4 && m == 0) ? "--------" : hex(m));
    }
    printf("\n");
  }
  cudaFree(d);
}

static void runRace(const char *mode) {
  int *d;
  CK(cudaMalloc(&d, 32 * sizeof(int)));
  if (strcmp(mode, "race-unsync") == 0)
    k_neighbor_unsync<<<1, 32>>>(d);
  else
    k_neighbor_sync<<<1, 32>>>(d);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  int h[32];
  CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < 32; ++i)
    if (h[i] != (i ^ 1) * 10) ok = false;
  printf("neighbour values logical-correct: %s\n", ok ? "yes" : "NO (torn)");
  for (int i = 0; i < 32; ++i) printf("%d ", h[i]);
  printf("\n");
  cudaFree(d);
}

static void runReduce(const char *mode) {
  int *d;
  CK(cudaMalloc(&d, 32 * sizeof(int)));
  if (strcmp(mode, "reduce-shfl") == 0)
    k_reduce_shfl<<<1, 32>>>(d);
  else if (strcmp(mode, "reduce-sync") == 0)
    k_reduce_sync<<<1, 32>>>(d);
  else
    k_reduce_nosync<<<1, 32>>>(d);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  int h[32];
  CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  printf("%-12s lane0=%d (want 496)  lane31=%d (not the sum; only lane0 is defined)\n",
         mode, h[0], h[31]);
  cudaFree(d);
}

static void runTiming() {
  const int n = 32;
  const long iters = 20000;
  unsigned long long *d;
  CK(cudaMalloc(&d, n * sizeof(*d)));
  k_timing<<<1, n>>>(d, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  unsigned long long h[32];
  CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  printf("lane : clock64 cycles (work scales with lane+1)\n");
  for (int i = 0; i < n; ++i) printf("%4d : %llu\n", i, h[i]);
  cudaFree(d);
}

static void runShfl(const char *mode) {
  int *d;
  CK(cudaMalloc(&d, 64 * sizeof(int)));
  CK(cudaMemset(d, 0, 64 * sizeof(int)));
  if (strcmp(mode, "shfl-bad") == 0) {
    k_shfl_bad<<<1, 32>>>(d);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    int h[64];
    CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    printf("lane : shfl result (want lane^1)*100 : observed mask\n");
    int bad = 0;
    for (int i = 0; i < 32; ++i) {
      int want = (i ^ 1) * 100;
      printf("%4d : want %4d  got %6d  mask %s%s\n", i, want, h[i],
             (i & 1) ? "(odd: did not shfl)" : hex((unsigned)h[32 + i]),
             (i % 2 == 0 && h[i] != want) ? "  <-- WRONG" : "");
      if (i % 2 == 0 && h[i] != want) bad++;
    }
    printf("even-lane mismatches: %d/16\n", bad);
  } else {
    k_shfl_good<<<1, 32>>>(d);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    int h[32];
    CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 32; ++i)
      if (h[i] != (i ^ 1) * 100) bad++;
    printf("shfl-good mismatches: %d/32\n", bad);
  }
  cudaFree(d);
}

static void runIntrinsics() {
  int *d, *ex;
  CK(cudaMalloc(&d, 32 * sizeof(int)));
  CK(cudaMalloc(&ex, (2 + 32) * sizeof(int)));
  k_intrinsics<<<1, 32>>>(d, ex);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  int h[32], he[34];
  CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(he, ex, sizeof(he), cudaMemcpyDeviceToHost));
  printf("warp reduce (shared, __syncwarp): lane0=%d (want 496)\n", h[0]);
  printf("__ballot_sync(0xffffffff, lane&1) = %s (want 0xaaaaaaaa)\n",
         hex((unsigned)he[0]));
  printf("__reduce_add_sync(0xffffffff, tid) = %d (want 496)\n", he[1]);
  printf("__match_any_sync(..., lane>>4):\n");
  for (int i = 0; i < 32; ++i) printf("  lane %2d -> %s\n", i, hex((unsigned)he[2 + i]));
  cudaFree(d);
  cudaFree(ex);
}

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "mask";
  if (mode == "mask") dumpMask();
  else if (mode == "race-unsync" || mode == "race-sync") runRace(mode.c_str());
  else if (mode == "shfl-bad" || mode == "shfl-good") runShfl(mode.c_str());
  else if (mode == "reduce-shfl" || mode == "reduce-nosync" ||
           mode == "reduce-sync") runReduce(mode.c_str());
  else if (mode == "timing") runTiming();
  else if (mode == "intrinsics") runIntrinsics();
  else if (mode == "sass") {
    int *d; int *o;
    CK(cudaMalloc(&d, 32 * sizeof(int)));
    CK(cudaMalloc(&o, 32 * sizeof(int)));
    k_sass<<<1, 32>>>(d, 1);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    printf("sass kernel ran\n");
    cudaFree(d); cudaFree(o);
  } else {
    printf("usage: %s {mask|race-unsync|race-sync|shfl-bad|shfl-good|"
           "reduce-shfl|reduce-nosync|reduce-sync|timing|intrinsics|sass}\n",
           argv[0]);
    return 2;
  }
  return 0;
}
