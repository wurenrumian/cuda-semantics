// Domain 11 — compiler behavior & undefined behavior (UB).
// Target: sm_89 (Ada, RTX 4060 Laptop), CUDA 12.8.
//
// Each demo probes one "promise boundary" of the CUDA C++ compiler:
//   counter-volatile  volatile read-modify-write is NOT atomic (lost updates).
//   counter-atomic    the atomic version is exact.
//   counter-shared    same race on shared memory -> needs __syncthreads + atomics.
//   restrict          __restrict__ tells the compiler p,q do not alias; lying is UB.
//   deadstore         overwritten / unused memory ops are deleted by the optimizer.
//   uninit            reading device memory never written is undefined.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t e = (x);                                                     \
    if (e != cudaSuccess) {                                                  \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,    \
             __LINE__);                                                      \
      exit(1);                                                              \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// 1) volatile is not atomic.
//    "volatile int" only stops the *compiler* from caching/folding the access.
//    It does NOT make "read-modify-write" atomic, and does NOT bring ordering.
// ---------------------------------------------------------------------------
__global__ void volatile_counter(int *counter, int iters) {
  volatile int *c = counter;      // volatile: access cannot be optimized away
  for (int i = 0; i < iters; ++i) {
    *c = *c + 1;                  // LD -> IADD -> ST: three separate steps
  }
}

__global__ void atomic_counter(int *counter, int iters) {
  for (int i = 0; i < iters; ++i) atomicAdd(counter, 1);
}

// Same idea on shared memory so that compute-sanitizer --tool racecheck sees it.
__global__ void volatile_shared_counter(int *out, int iters) {
  __shared__ int c;
  if (threadIdx.x == 0) c = 0;
  __syncthreads();
  volatile int *vc = &c;
  for (int i = 0; i < iters; ++i) {
    *vc = *vc + 1;                // data race on shared c (no atomic)
  }
  __syncthreads();
  if (threadIdx.x == 0) out[0] = c;
}

// ---------------------------------------------------------------------------
// 2) __restrict__ / aliasing.
//    __restrict__ promises the two pointers do not alias. The compiler is then
//    allowed to reorder the store and the load. If the promise is a lie
//    (p == q), the result is undefined; here we feed it a lie on purpose.
// ---------------------------------------------------------------------------
__global__ void store_then_load_restrict(int *__restrict__ p,
                                         int *__restrict__ q, int *out) {
  q[0] = 42;                      // store through q
  int a = p[0];                   // load through p; may be hoisted above the store
  out[0] = a;
}

__global__ void store_then_load_plain(int *p, int *q, int *out) {
  q[0] = 42;                      // p and q may alias, so the load must be ordered
  int a = p[0];
  out[0] = a;
}

// ---------------------------------------------------------------------------
// 3) Dead-store / unused-access elimination.
//    A memory write that no one can observe is not required to happen at all.
//    volatile forbids that elimination; plane pointers allow it.
// ---------------------------------------------------------------------------
__global__ void dead_store(int *out, const int *in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int x = in[i];
  out[i] = x;                     // written ...
  out[i] = x + 1;                 // ... then immediately overwritten -> first store is dead
}

__global__ void dead_store_volatile(volatile int *out, const int *in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int x = in[i];
  out[i] = x;                     // volatile: both stores must be emitted
  out[i] = x + 1;
}

__global__ void unused_load(int *out, const int *in) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int x = in[i];                  // loaded ...
  out[i] = 7;                     // ... but never used -> load not emitted
}

// ---------------------------------------------------------------------------
// 4) Uninitialized device memory read.
//    cudaMalloc does not zero memory. Reading it is undefined; initcheck flags it.
// ---------------------------------------------------------------------------
__global__ void read_uninit(int *out, const int *buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = buf[i] * 2; // buf was never written
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "help";

  if (mode == "counter-volatile") {
    const int threads = 256, iters = 200000;
    long long total = (long long)threads * iters;
    int *d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
    CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
    volatile_counter<<<1, threads>>>(d, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int h = 0;
    CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
    printf("volatile counter : expected=%lld  observed=%d  lost=%lld\n",
           total, h, total - h);
    cudaFree(d);

  } else if (mode == "counter-atomic") {
    const int threads = 256, iters = 200000;
    long long total = (long long)threads * iters;
    int *d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
    CUDA_CHECK(cudaMemset(d, 0, sizeof(int)));
    atomic_counter<<<1, threads>>>(d, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int h = 0;
    CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
    printf("atomic   counter : expected=%lld  observed=%d  lost=%lld\n",
           total, h, total - h);
    cudaFree(d);

  } else if (mode == "counter-shared") {
    const int threads = 256, iters = 20000;
    int *d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
    volatile_shared_counter<<<1, threads>>>(d, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int h = 0;
    CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
    printf("shared counter   : expected=%lld  observed=%d\n",
           (long long)threads * iters, h);
    cudaFree(d);

  } else if (mode == "restrict") {
    int *buf, *out;
    CUDA_CHECK(cudaMalloc(&buf, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out, sizeof(int)));
    int seven = 7;
    CUDA_CHECK(cudaMemcpy(buf, &seven, sizeof(int), cudaMemcpyHostToDevice));

    // p == q : the __restrict__ promise is violated (undefined behaviour).
    CUDA_CHECK(cudaMemset(out, 0, sizeof(int)));
    store_then_load_restrict<<<1, 1>>>(buf, buf, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int hr = 0;
    CUDA_CHECK(cudaMemcpy(&hr, out, sizeof(int), cudaMemcpyDeviceToHost));

    // Same aliasing, but pointers are not marked __restrict__.
    CUDA_CHECK(cudaMemcpy(buf, &seven, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(out, 0, sizeof(int)));
    store_then_load_plain<<<1, 1>>>(buf, buf, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int hp = 0;
    CUDA_CHECK(cudaMemcpy(&hp, out, sizeof(int), cudaMemcpyDeviceToHost));

    printf("aliased store q[0]=42 then read p[0] (p==q, buf started at 7)\n");
    printf("  __restrict__ version -> out = %d   (7 == load hoisted above store)\n", hr);
    printf("  plain        version -> out = %d   (42 == store observed)\n", hp);
    cudaFree(buf); cudaFree(out);

  } else if (mode == "deadstore") {
    const int n = 4;
    int *in, *out;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    int hi[4] = {10, 11, 12, 13};
    CUDA_CHECK(cudaMemcpy(in, hi, sizeof(hi), cudaMemcpyHostToDevice));
    dead_store<<<1, n>>>(out, in);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int ho[4] = {0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(ho, out, sizeof(ho), cudaMemcpyDeviceToHost));
    printf("dead_store result: %d %d %d %d (only the surviving store matters)\n",
           ho[0], ho[1], ho[2], ho[3]);
    cudaFree(in); cudaFree(out);

  } else if (mode == "uninit") {
    const int n = 4;
    int *buf, *out;
    CUDA_CHECK(cudaMalloc(&buf, n * sizeof(int)));   // never written
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    read_uninit<<<1, n>>>(out, buf, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int ho[4] = {0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(ho, out, sizeof(ho), cudaMemcpyDeviceToHost));
    printf("read_uninit result: %d %d %d %d (values are undefined)\n",
           ho[0], ho[1], ho[2], ho[3]);
    cudaFree(buf); cudaFree(out);

  } else {
    printf("usage: %s {counter-volatile|counter-atomic|counter-shared|restrict|deadstore|uninit}\n",
           argv[0]);
    return 2;
  }
  return 0;
}
