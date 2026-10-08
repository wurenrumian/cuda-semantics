// EXP-0009 — Asynchronous execution & pipelines (domain 09).
// Target: sm_89 (Ada, RTX 4060 Laptop), CUDA 12.8.  Build & run: scripts/run.ps1
//
// Modes:
//   props        — device properties relevant to async/streams semantics
//   streams      — same-stream serialization vs cross-stream concurrency
//   ds           — null-stream ordering: legacy vs per-thread default stream
//                  (same source is compiled twice with -default-stream ...)
//   memcpy       — cudaMemcpyAsync return time: pageable vs pinned host memory
//   cpasync      — cp.async (__pipeline_memcpy_async) -> shared, correctness
//   cpasync-racy — same without __pipeline_wait_prior (a data race / UB)
//   graph        — CUDA Graph capture/replay execution semantics
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,      \
             __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch())
      .count();
}

// A compute-bound kernel occupying one SM (grid=1). The LCG cannot be folded.
__global__ void busy_kernel(long iters, int *sink) {
  unsigned long long s =
      (unsigned long long)(threadIdx.x + blockIdx.x * blockDim.x) + 1;
  for (long i = 0; i < iters; ++i)
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  if (s == 0xdeadbeefdeadbeefULL) sink[0] = 1;
}

// ---------------------------------------------------------------------------
static void mode_props() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("name                       : %s\n", p.name);
  printf("compute capability         : %d.%d\n", p.major, p.minor);
  printf("multiProcessorCount        : %d\n", p.multiProcessorCount);
  printf("warpSize                   : %d\n", p.warpSize);
  printf("asyncEngineCount           : %d\n", p.asyncEngineCount);
  printf("concurrentKernels          : %d\n", p.concurrentKernels);
  printf("canMapHostMemory           : %d\n", p.canMapHostMemory);
  printf("unifiedAddressing          : %d\n", p.unifiedAddressing);
  printf("managedMemory              : %d\n", p.managedMemory);
  printf("pageableMemoryAccess       : %d\n", p.pageableMemoryAccess);
  printf("directManagedMemAccessFromHost: %d\n",
         p.directManagedMemAccessFromHost);
  printf("l2CacheSize                : %d\n", p.l2CacheSize);
  printf("cudaStreamPerThread        : %d (non-NULL on sm_70+)\n",
         cudaStreamPerThread ? 1 : 0);
}

// ---------------------------------------------------------------------------
static void mode_streams() {
  long iters = 60000000;
  int n = 4;
  int *sink;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));
  CUDA_CHECK(cudaMemset(sink, 0, sizeof(int)));

  // warm up
  busy_kernel<<<1, 256>>>(iters / 4, sink);
  CUDA_CHECK(cudaDeviceSynchronize());

  // all 4 on the same stream -> serialized
  double a = now_ms();
  for (int i = 0; i < n; ++i) busy_kernel<<<1, 256>>>(iters, sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  double same = now_ms() - a;

  // one per stream -> may run concurrently
  cudaStream_t st[4];
  for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaStreamCreate(&st[i]));
  double b = now_ms();
  for (int i = 0; i < n; ++i)
    busy_kernel<<<1, 256, 0, st[i % 4]>>>(iters, sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  double multi = now_ms() - b;

  printf("one busy kernel         : ~%.1f ms\n", same / n);
  printf("4 kernels, 1 stream     : %.1f ms  (serialized)\n", same);
  printf("4 kernels, 4 streams    : %.1f ms  (concurrency if < 4x)\n", multi);
  printf("speedup (1 stream / 4)  : %.2fx\n", same / multi);

  for (int i = 0; i < 4; ++i) cudaStreamDestroy(st[i]);
  cudaFree(sink);
}

// ---------------------------------------------------------------------------
// Compiled twice: -default-stream legacy (default) and per-thread.
static void mode_ds() {
  long iters = 80000000;
  int *sink;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));
  CUDA_CHECK(cudaMemset(sink, 0, sizeof(int)));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // reference: one kernel alone
  double r = now_ms();
  busy_kernel<<<1, 256, 0, s>>>(iters, sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  double one = now_ms() - r;

  // kernel on explicit stream s, then kernel on the null (default) stream
  double t = now_ms();
  busy_kernel<<<1, 256, 0, s>>>(iters, sink);
  busy_kernel<<<1, 256, 0, 0>>>(iters, sink); // null / default stream
  CUDA_CHECK(cudaDeviceSynchronize());
  double two = now_ms() - t;

  printf("one busy kernel on stream s      : ~%.1f ms\n", one);
  printf("s-kernel + null-stream kernel    : %.1f ms\n", two);
  printf("ratio (should be ~1 if overlapped, ~2 if serialized): %.2f\n",
         two / one);
  cudaStreamDestroy(s);
  cudaFree(sink);
}

// ---------------------------------------------------------------------------
static void mode_memcpy() {
  size_t bytes = 256ull << 20; // 256 MiB
  void *hpage = malloc(bytes);
  void *hpin = nullptr;
  void *d = nullptr;
  CUDA_CHECK(cudaHostAlloc(&hpin, bytes, cudaHostAllocDefault));
  CUDA_CHECK(cudaMalloc(&d, bytes));
  memset(hpage, 1, bytes);
  memset(hpin, 1, bytes);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // warm up + reference synchronous copy time
  CUDA_CHECK(cudaMemcpyAsync(d, hpin, bytes, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  double r = now_ms();
  CUDA_CHECK(cudaMemcpy(d, hpin, bytes, cudaMemcpyHostToDevice));
  double sync_ms = now_ms() - r;

  // pageable host memory: cudaMemcpyAsync blocks until staging copy is done
  double t1 = now_ms();
  CUDA_CHECK(cudaMemcpyAsync(d, hpage, bytes, cudaMemcpyHostToDevice, s));
  double pageable = now_ms() - t1;
  CUDA_CHECK(cudaStreamSynchronize(s));

  // pinned host memory: returns immediately, transfer happens async
  double t2 = now_ms();
  CUDA_CHECK(cudaMemcpyAsync(d, hpin, bytes, cudaMemcpyHostToDevice, s));
  double pinned = now_ms() - t2;
  CUDA_CHECK(cudaStreamSynchronize(s));

  printf("payload                       : %zu MiB\n", bytes >> 20);
  printf("cudaMemcpy (sync)             : %.2f ms\n", sync_ms);
  printf("cudaMemcpyAsync return, pageable: %.2f ms  <- blocks/stages on host\n",
         pageable);
  printf("cudaMemcpyAsync return, pinned  : %.3f ms <- returns immediately\n",
         pinned);

  cudaStreamDestroy(s);
  cudaFree(d);
  cudaFreeHost(hpin);
  free(hpage);
}

// ---------------------------------------------------------------------------
// cp.async: global -> shared without going through registers (sm_80+).
__global__ void cpasync_copy(const float4 *__restrict__ g,
                             float4 *__restrict__ out, int n4) {
  extern __shared__ float4 sh[];
  int t = threadIdx.x;
  if (t < n4) __pipeline_memcpy_async(&sh[t], &g[t], sizeof(float4));
  __pipeline_commit();
  __pipeline_wait_prior(0); // data must land before it is read
  __syncthreads();
  if (t < n4) out[t] = sh[t];
}

// Intentionally omit wait_prior: reading shared before cp.async completes is a
// data race (UB) — the same shared buffer is read before the async write lands.
__global__ void cpasync_racy(const float4 *__restrict__ g,
                             float4 *__restrict__ out, int n4) {
  extern __shared__ float4 sh[];
  int t = threadIdx.x;
  if (t < n4) __pipeline_memcpy_async(&sh[t], &g[t], sizeof(float4));
  __pipeline_commit();
  // __pipeline_wait_prior(0) intentionally missing
  __syncthreads();
  if (t < n4) out[t] = sh[t];
}

// Plain path for SASS comparison: LDG -> registers -> STS.
__global__ void manual_copy(const float4 *__restrict__ g,
                            float4 *__restrict__ out, int n4) {
  extern __shared__ float4 sh[];
  int t = threadIdx.x;
  if (t < n4) sh[t] = g[t];
  __syncthreads();
  if (t < n4) out[t] = sh[t];
}

static int run_copy(int which /*0 async,1 manual,2 racy*/, int reps) {
  const int n4 = 256;
  float4 *g, *out;
  CUDA_CHECK(cudaMalloc(&g, n4 * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&out, n4 * sizeof(float4)));
  float4 *hg = (float4 *)malloc(n4 * sizeof(float4));
  float4 *ho = (float4 *)malloc(n4 * sizeof(float4));
  for (int i = 0; i < n4; ++i)
    hg[i] = make_float4((float)i, (float)i + 1, (float)i + 2, (float)i + 3);
  CUDA_CHECK(cudaMemcpy(g, hg, n4 * sizeof(float4), cudaMemcpyHostToDevice));

  int bad = 0;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaMemset(out, 0, n4 * sizeof(float4)));
    size_t smem = n4 * sizeof(float4);
    if (which == 0)
      cpasync_copy<<<1, n4, smem>>>(g, out, n4);
    else if (which == 1)
      manual_copy<<<1, n4, smem>>>(g, out, n4);
    else
      cpasync_racy<<<1, n4, smem>>>(g, out, n4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ho, out, n4 * sizeof(float4), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n4; ++i)
      if (!(ho[i].x == hg[i].x && ho[i].y == hg[i].y && ho[i].z == hg[i].z &&
            ho[i].w == hg[i].w))
        ++bad;
  }
  const char *name = which == 0 ? "cpasync(correct)" :
                     which == 1 ? "manual" : "cpasync-racy(no wait)";
  printf("%-24s : %d/%d element mismatches over %d reps\n", name, bad,
         n4 * reps, reps);
  cudaFree(g);
  cudaFree(out);
  free(hg);
  free(ho);
  return bad;
}

static void mode_cpasync(int which) {
  if (which == 2) {
    // run several times; racy behavior may be intermittent
    for (int k = 0; k < 5; ++k) run_copy(2, 1);
  } else {
    run_copy(1, 1);
    run_copy(0, 1);
  }
}

// ---------------------------------------------------------------------------
__global__ void inc_kernel(int *d) { d[0] = d[0] + 1; }

static void mode_graph() {
  int *d;
  CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // (a) capturing the legacy default stream is not allowed
  cudaGraph_t gtmp = nullptr;
  cudaError_t e = cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
  printf("cudaStreamBeginCapture(null stream) -> %s (%s)\n",
         cudaGetErrorName(e), cudaGetErrorString(e));
  if (e == cudaSuccess) cudaStreamEndCapture(0, &gtmp);

  // (b) capture a 2-launch sequence on a non-default stream
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  inc_kernel<<<1, 1, 0, s>>>(d);
  inc_kernel<<<1, 1, 0, s>>>(d);
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));

  int h = -1;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
  printf("after capture, counter = %d  (capture records, does not run)\n", h);

  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

  CUDA_CHECK(cudaGraphLaunch(exec, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
  printf("after replay #1, counter = %d\n", h);

  CUDA_CHECK(cudaGraphLaunch(exec, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
  printf("after replay #2, counter = %d\n", h);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(s);
  cudaFree(d);
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  std::string m = argc > 1 ? argv[1] : "props";
  if (m == "props")        mode_props();
  else if (m == "streams") mode_streams();
  else if (m == "ds")      mode_ds();
  else if (m == "memcpy")  mode_memcpy();
  else if (m == "cpasync") mode_cpasync(0);
  else if (m == "cpasync-racy") mode_cpasync(2);
  else if (m == "graph")   mode_graph();
  else { printf("usage: %s {props|streams|ds|memcpy|cpasync|cpasync-racy|graph}\n",
                argv[0]); return 2; }
  return 0;
}
