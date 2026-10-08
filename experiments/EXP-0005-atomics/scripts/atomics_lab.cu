// D05 atomics lab -- runtime functional demos.
// Target: sm_89 (Ada, RTX 4060 Laptop) / CUDA 12.8.  Build & run: run.ps1
//
// Modes:
//   props        -- device facts relevant to atomics
//   atomicity    -- non-atomic RMW loses updates; atomicAdd does not
//   race-naive   -- shared counter without atomics (racecheck target: race)
//   race-atomic  -- same counter with atomicAdd  (racecheck target: clean)
//   publish <0|1> <rounds>  -- flag+payload message passing, fence or no fence
//   floatmax     -- the "cast float to int and atomicMax" trick, sign artifact
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                        \
    if (e != cudaSuccess) {                                                     \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);    \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

__device__ int g_data[4096];
__device__ int g_ready;
__device__ int g_counter;

// -------------------- atomicity -------------------------------------------
__global__ void global_inc_naive(volatile int *x, int iters) {
  for (int i = 0; i < iters; ++i) { int v = *x; *x = v + 1; }
}
__global__ void global_inc_atomic(int *x, int iters) {
  for (int i = 0; i < iters; ++i) atomicAdd(x, 1);
}

// -------------------- shared-memory race (racecheck) ----------------------
__global__ void shared_counter_naive(int *out, int iters) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  for (int i = 0; i < iters; ++i) { int v = s; s = v + 1; }
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}
__global__ void shared_counter_atomic(int *out, int iters) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  for (int i = 0; i < iters; ++i) atomicAdd(&s, 1);
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// -------------------- publish (flag + payload) ----------------------------
__global__ void pub_producer(int useFence, int n, int *data, int *ready) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; ++i) data[i] = i + 1;     // payload (plain stores)
    if (useFence) __threadfence();                   // release
    atomicExch(ready, 1);                            // relaxed signal
  }
}
__global__ void pub_consumer(int useFence, int n, int *data, int *ready, int *bad) {
  if (threadIdx.x == 0) {
    while (atomicAdd(ready, 0) == 0) { }             // relaxed spin
    if (useFence) __threadfence();                   // acquire
    int b = 0;
    for (int i = 0; i < n; ++i)
      if (data[i] != i + 1) ++b;
    *bad = b;
  }
}

// -------------------- float atomicMax via int bits ------------------------
__global__ void naive_fmax(float *out, float v) {
  if (threadIdx.x == 0) atomicMax((int *)out, __float_as_int(v));
}
__global__ void proper_fmax(float *out, float v) {
  if (threadIdx.x == 0) {
    int *p = (int *)out;
    int old = *p, assumed;
    do {
      assumed = old;
      old = atomicCAS(p, assumed, __float_as_int(fmaxf(v, __int_as_float(assumed))));
    } while (assumed != old);
  }
}

static void mode_props() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("name                 : %s\n", p.name);
  printf("compute capability   : %d.%d\n", p.major, p.minor);
  printf("warpSize             : %d\n", p.warpSize);
  printf("multiProcessorCount  : %d\n", p.multiProcessorCount);
  printf("l2CacheSize          : %d\n", p.l2CacheSize);
  printf("concurrentKernels    : %d\n", p.concurrentKernels);
}

static void mode_atomicity() {
  const int n = 256, iters = 2000;
  const long expect = (long)n * iters;
  int *d; CK(cudaMalloc(&d, sizeof(int)));

  CK(cudaMemset(d, 0, sizeof(int)));
  global_inc_naive<<<1, n>>>((volatile int *)d, iters); CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  int a; CK(cudaMemcpy(&a, d, sizeof(int), cudaMemcpyDeviceToHost));

  CK(cudaMemset(d, 0, sizeof(int)));
  global_inc_atomic<<<1, n>>>(d, iters); CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  int b; CK(cudaMemcpy(&b, d, sizeof(int), cudaMemcpyDeviceToHost));

  printf("expected                         : %ld\n", expect);
  printf("non-atomic  (*x = *x + 1)        : %d  (lost %ld)\n", a, expect - a);
  printf("atomicAdd(*x, 1)                 : %d  (lost %ld)\n", b, expect - b);
  cudaFree(d);
}

static void mode_race(bool atomic) {
  const int iters = 256;
  int *d; CK(cudaMalloc(&d, sizeof(int)));
  if (atomic) shared_counter_atomic<<<1, 32>>>(d, iters);
  else        shared_counter_naive<<<1, 32>>>(d, iters);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  int h; CK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
  printf("%s shared counter: %d (expected %d)\n",
         atomic ? "atomic" : "non-atomic", h, 32 * iters);
  cudaFree(d);
}

static void mode_publish(int useFence, int rounds) {
  const int n = 1024;
  int *d, *data, *ready;
  CK(cudaMalloc(&d, sizeof(int)));
  CK(cudaMalloc(&data, sizeof(int) * n));
  CK(cudaMalloc(&ready, sizeof(int)));
  cudaStream_t s1, s2; CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s2));
  long badRounds = 0, maxBad = 0;
  for (int r = 0; r < rounds; ++r) {
    CK(cudaMemset(data, 0, sizeof(int) * n));
    CK(cudaMemset(ready, 0, sizeof(int)));
    CK(cudaDeviceSynchronize());
    pub_producer<<<1, 32, 0, s1>>>(useFence, n, data, ready);
    pub_consumer<<<1, 32, 0, s2>>>(useFence, n, data, ready, d);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int bad; CK(cudaMemcpy(&bad, d, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad > 0) { ++badRounds; if (bad > maxBad) maxBad = bad; }
  }
  printf("publish fence=%d rounds=%d : badRounds=%ld maxBadInts=%ld\n",
         useFence, rounds, badRounds, maxBad);
  cudaStreamDestroy(s1); cudaStreamDestroy(s2);
  cudaFree(d); cudaFree(data); cudaFree(ready);
}

static void mode_floatmax() {
  float *d; CK(cudaMalloc(&d, sizeof(float)));
  struct Case { float init, val; const char *name; };
  Case cases[] = {
    {-5.0f, -3.0f, "both negative  (trick breaks)"},
    { 3.0f,  5.0f, "both positive  (trick works)"},
    {-3.0f,  5.0f, "mixed sign     (trick works)"},
  };
  for (Case c : cases) {
    CK(cudaMemcpy(d, &c.init, sizeof(float), cudaMemcpyHostToDevice));
    naive_fmax<<<1, 1>>>(d, c.val); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float naive; CK(cudaMemcpy(&naive, d, sizeof(float), cudaMemcpyDeviceToHost));

    CK(cudaMemcpy(d, &c.init, sizeof(float), cudaMemcpyHostToDevice));
    proper_fmax<<<1, 1>>>(d, c.val); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float proper; CK(cudaMemcpy(&proper, d, sizeof(float), cudaMemcpyDeviceToHost));

    float want = c.val > c.init ? c.val : c.init;
    printf("init=%6.1f max(%6.1f): naive=%6.1f (%s)  proper=%6.1f (%s)  want=%6.1f\n",
           c.init, c.val, naive, naive == want ? "ok" : "WRONG",
           proper, proper == want ? "ok" : "WRONG", want);
  }
  cudaFree(d);
}

int main(int argc, char **argv) {
  std::string m = argc > 1 ? argv[1] : "props";
  if      (m == "props")     mode_props();
  else if (m == "atomicity") mode_atomicity();
  else if (m == "race-naive")  mode_race(false);
  else if (m == "race-atomic") mode_race(true);
  else if (m == "publish")   mode_publish(argc > 2 ? atoi(argv[2]) : 0,
                                          argc > 3 ? atoi(argv[3]) : 200);
  else if (m == "floatmax")  mode_floatmax();
  else { printf("usage: %s {props|atomicity|race-naive|race-atomic|publish <0|1> <rounds>|floatmax}\n", argv[0]); return 2; }
  return 0;
}
