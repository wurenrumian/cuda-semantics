// EXP-0003 — CUDA memory spaces: where a variable actually lives, and what each
// space promises about scope / lifetime / visibility.
// Target: sm_89 (Ada, RTX 4060 Laptop), CUDA 12.8.  Build & run: scripts/run.ps1
//
// Modes: attrs | local | chase | const | shared | coalesce | readonly | all
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,      \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ===========================================================================
// A) "local" is a compiler decision, not a source declaration.
//    k_local_promoted : `float arr[16]`, masked (provably in-range) run-time
//                       index -> ptxas promotes the whole array into registers
//                       (0 LDL/STL in SASS).
//    k_local_loop     : same array, declared volatile, indexed in a long loop
//                       -> ptxas must keep it in local memory (LDL/STL).
// ===========================================================================
__global__ void k_local_promoted(const float *in, float *out, int a, int b) {
  float arr[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) arr[i] = in[i];
  arr[a & 15] = in[16] + 1.0f;
  out[threadIdx.x] = arr[b & 15];
}

__global__ void k_local_loop(const float *in, float *out, int sel, long reps) {
  volatile float arr[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) arr[i] = in[i];
  float acc = 0.f;
  int j = sel & 15;
  for (long r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      acc += arr[j];
      j = (j + 1) & 15;
    }
  }
  out[threadIdx.x] = acc;
}

// B) load-to-use latency through a dependent index (pointer chasing):
//    local memory (device memory, L1-cached) vs shared memory (on-chip SRAM).
//    The index map is j -> (7*j) mod 16. gcd(7,16)=1 and 7^2==1 (mod 16), so the
//    map is a permutation / involution: the chase settles into a fixed point (0 or 8)
//    or a 2-cycle. It never diverges, and the loop is bounded by `reps`, so the chain
//    always runs exactly `reps` dependent loads and then exits.
__global__ void k_local_chase(const float *in, float *out, long reps) {
  volatile float arr[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) arr[i] = (float)((i * 7) & 15);
  int j = (int)in[15] & 15;
  for (long r = 0; r < reps; ++r) j = ((int)arr[j]) & 15;
  out[threadIdx.x] = j;
}

__global__ void k_shared_chase(const float *in, float *out, long reps) {
  __shared__ float s[16];              // same index map as k_local_chase: j -> 7j mod 16
  if (threadIdx.x < 16) s[threadIdx.x] = (float)((threadIdx.x * 7) & 15);
  __syncthreads();
  int j = (int)in[15] & 15;
  for (long r = 0; r < reps; ++r) j = ((int)s[j]) & 15;
  out[threadIdx.x] = j;
}

// ===========================================================================
// C) constant memory: uniform (broadcast) vs divergent address.
//    Both do ONE read per iteration; only the address pattern differs.
// ===========================================================================
__constant__ float cbuf[128];

__global__ void k_const_broadcast(float *out, long reps) {
  float acc = 0.f;
  for (long r = 0; r < reps; ++r) acc += cbuf[(int)(r * 7) & 127];
  out[threadIdx.x] = acc;
}

__global__ void k_const_divergent(float *out, long reps) {
  int lane = threadIdx.x & 31;
  float acc = 0.f;
  for (long r = 0; r < reps; ++r) acc += cbuf[(int)(r * 7 + lane) & 127];
  out[threadIdx.x] = acc;
}

// ===========================================================================
// D) shared memory bank conflict: address = lane*stride + r.
//    stride 1  -> distinct banks           -> conflict-free.
//    stride 32 -> bank = r for every lane  -> 32-way conflict.
// ===========================================================================
__global__ void k_shared(float *out, int stride, long reps) {
  __shared__ float s[1024];
  int tid = threadIdx.x;
  for (int i = tid; i < 1024; i += blockDim.x) s[i] = (float)i;
  __syncthreads();
  int idx = tid & 31;
  float acc = 0.f;
  for (long r = 0; r < reps; ++r) acc += s[(idx * stride + (int)r) & 1023];
  out[tid] = acc;
}

// ===========================================================================
// E) global coalescing: contiguous (stride 1) vs strided (stride 32 floats).
//    Many warps so the measurement is bandwidth-bound, not latency-bound.
//    `r*97` decorrelates successive iterations; `tid*stride` fixes the warp access
//    pattern: stride 1 -> consecutive floats (one 128B line per warp), stride 32 ->
//    128B apart per lane (32 distinct sectors per warp). `& mask` keeps every read
//    address in bounds; the loop count is bounded by `reps`. The write
//    `out[tid] = acc` needs one slot per thread -- the host sizes `out` for the
//    full 65536-thread grid (see main).
// ===========================================================================
__global__ void k_coal(const float *in, float *out, int stride, long reps,
                       int mask) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  for (long r = 0; r < reps; ++r)
    acc += in[(int)((r * 97 + tid * stride) & mask)];
  out[tid] = acc;
}

// ===========================================================================
// F) read-only path: __ldg -> ld.global.nc -> LDG.E.CONSTANT in SASS.
// ===========================================================================
__global__ void k_readonly(const float *__restrict__ in, float *out) {
  out[threadIdx.x] = __ldg(&in[threadIdx.x]) + in[threadIdx.x + 32];
}

// ===========================================================================
// G) local memory is a real, bounded, per-thread region: an out-of-bounds index
//    is caught by compute-sanitizer (memcheck) as an invalid __local__ access.
//    __noinline__ makes the array's address escape, so ptxas cannot promote it.
// ===========================================================================
__device__ __noinline__ float opaque_read(const float *p, int i) { return p[i]; }

__global__ void k_local_oob(const float *in, float *out, int n) {
  volatile float arr[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) arr[i] = in[i];
  out[threadIdx.x] = opaque_read((const float *)arr, n);
}

// ---------------------------------------------------------------------------
template <class F>
static float time_kernel(F launch, long reps, int warmup = 1000) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  launch(warmup);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(e0));
  launch(reps);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms;
}

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "all";

  std::vector<float> h(4096);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (float)(i % 97) * 0.5f;
  // dout must hold one element per active thread of the LARGEST launch:
  // k_coal runs 256x256 = 65536 threads and every thread stores out[tid]; the
  // other kernels use <= 64 threads. (Previously dout was sized to h only,
  // which sent k_coal's writes ~236 KB past the end of the allocation.)
  const size_t outElems = std::max(h.size(), (size_t)65536);
  float *din = nullptr, *dout = nullptr;
  CUDA_CHECK(cudaMalloc(&din, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, outElems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(din, h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  if (mode == "attrs" || mode == "all") {
    cudaFuncAttributes p, l, c, sc;
    CUDA_CHECK(cudaFuncGetAttributes(&p, k_local_promoted));
    CUDA_CHECK(cudaFuncGetAttributes(&l, k_local_loop));
    CUDA_CHECK(cudaFuncGetAttributes(&c, k_local_chase));
    CUDA_CHECK(cudaFuncGetAttributes(&sc, k_shared_chase));
    printf("%-20s %6s %14s\n", "kernel", "regs", "localSizeBytes");
    printf("%-20s %6d %14llu\n", "k_local_promoted", p.numRegs,
           (unsigned long long)p.localSizeBytes);
    printf("%-20s %6d %14llu\n", "k_local_loop", l.numRegs,
           (unsigned long long)l.localSizeBytes);
    printf("%-20s %6d %14llu\n", "k_local_chase", c.numRegs,
           (unsigned long long)c.localSizeBytes);
    printf("%-20s %6d %14llu\n", "k_shared_chase", sc.numRegs,
           (unsigned long long)sc.localSizeBytes);
  }

  if (mode == "local" || mode == "all") {
    const long reps = 200000;
    k_local_promoted<<<1, 32>>>(din, dout, 5, 9);
    k_local_loop<<<1, 32>>>(din, dout, 3, 100);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms =
        time_kernel([&](long r) { k_local_loop<<<1, 32>>>(din, dout, 3, r); },
                    reps);
    printf("volatile local array loop : %8.3f ms  (%ld iters x 16 LDL)\n", ms,
           reps);
  }

  if (mode == "chase" || mode == "all") {
    const long reps = 2000000;
    float ms_l =
        time_kernel([&](long r) { k_local_chase<<<1, 32>>>(din, dout, r); },
                    reps);
    float ms_s =
        time_kernel([&](long r) { k_shared_chase<<<1, 32>>>(din, dout, r); },
                    reps);
    printf("dependent-index chase, local (LDL)  : %8.3f ms  (%ld hops)\n", ms_l,
           reps);
    printf("dependent-index chase, shared (LDS) : %8.3f ms  (%ld hops)\n", ms_s,
           reps);
    printf("local/shared latency ratio = %.2fx\n", ms_l / ms_s);
  }

  if (mode == "const" || mode == "all") {
    std::vector<float> c(128);
    for (int i = 0; i < 128; ++i) c[i] = (float)i;
    CUDA_CHECK(cudaMemcpyToSymbol(cbuf, c.data(), c.size() * sizeof(float)));
    const long reps = 1000000;
    float ms_b = time_kernel(
        [&](long r) { k_const_broadcast<<<1, 32>>>(dout, r); }, reps);
    float ms_d = time_kernel(
        [&](long r) { k_const_divergent<<<1, 32>>>(dout, r); }, reps);
    printf("const broadcast (uniform addr) : %8.3f ms  (%ld reads)\n", ms_b,
           reps);
    printf("const divergent (32 per warp)  : %8.3f ms  (%ld reads/lane)\n", ms_d,
           reps);
    printf("divergent/broadcast ratio = %.2fx\n", ms_d / ms_b);
  }

  if (mode == "shared" || mode == "all") {
    const long reps = 300000;
    for (int stride : {1, 32}) {
      float ms = time_kernel(
          [&](long r) { k_shared<<<1, 32>>>(dout, stride, r); }, reps);
      printf("shared stride=%2d : %8.3f ms  (%ld iters x 32 lanes)\n", stride, ms,
             reps);
    }
  }

  if (mode == "coalesce" || mode == "all") {
    // Working set far larger than cache so the access pattern dominates.
    const int nbig = 1 << 23; // 8M floats = 32 MB
    const int mask = nbig - 1;
    float *big = nullptr;
    CUDA_CHECK(cudaMalloc(&big, (size_t)nbig * sizeof(float)));
    CUDA_CHECK(cudaMemset(big, 0, (size_t)nbig * sizeof(float)));
    const long reps = 4000;
    for (int stride : {1, 32}) {
      float ms = time_kernel([&](long r) {
        k_coal<<<256, 256>>>(big, dout, stride, r, mask);
      }, reps);
      printf("global stride=%2d : %8.3f ms  (%ld iters, 65536 threads, 32MB)\n",
             stride, ms, reps);
    }
    cudaFree(big);
  }

  if (mode == "oob") {
    // n = 100 is far outside the 8-element local array.
    k_local_oob<<<1, 32>>>(din, dout, 100);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("oob kernel ran\n");
  }

  if (mode == "readonly" || mode == "all") {
    k_readonly<<<1, 64>>>(din, dout);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("readonly kernel ran (see SASS: LDG.E.CONSTANT)\n");
  }

  cudaFree(din);
  cudaFree(dout);
  return 0;
}
