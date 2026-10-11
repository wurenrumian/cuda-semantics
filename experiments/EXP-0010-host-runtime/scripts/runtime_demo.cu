// EXP-0010 — host-side runtime / launch semantics.
// Target: sm_89 (Ada, RTX 4060 Laptop), CUDA 12.8.  Build & run: scripts/run.ps1
//
// Modes (each run as a FRESH process — an async fault can poison the context):
//   config   — invalid launch configuration is reported SYNCHRONOUSLY.
//   peekget  — cudaPeekAtLastError does NOT clear; cudaGetLastError does.
//   async    — an execution fault is ASYNCHRONOUS: launch looks fine, it blows
//              up only at the next synchronization point. (the gotcha)
//   streams  — legacy default stream implicitly serializes with BLOCKING streams.
//   props    — print the launch-related device limits on this machine.
//
// Build: vcvars64.bat ; nvcc -O3 -lineinfo -arch=sm_89 -o runtime_demo.exe runtime_demo.cu
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__global__ void nop() {}

// Deliberate out-of-bounds / null dereference: a device execution fault.
__global__ void deref_null(int *p) { *p = 123; }

__global__ void set_flag(volatile int *f) { *f = 1; }

// Spin until *flag != 0, or until a wall-clock budget expires.
// The budget is measured with clock64() (the SM clock), so the spin is bounded
// in TIME rather than in iteration count: a spin that is never woken must time
// out quickly instead of hanging (the "blocking" case below relies on exactly
// that timeout).  Writes the last observed value of *flag: 1 if set_flag won
// the race, 0 if the wait gave up.
__global__ void spin_wait(volatile int *flag, int *out,
                          long long timeout_cycles) {
  long long start = clock64();
  while (*flag == 0 && (clock64() - start) < timeout_cycles) {}
  *out = (int)(*flag);
}

static void report(const char *label, cudaError_t e) {
  printf("  %-34s : %s (%d)\n", label, cudaGetErrorString(e), (int)e);
}

// ---------------------------------------------------------------------------
static void mode_config() {
  printf("== invalid launch config -> SYNCHRONOUS error at the call site ==\n");
  nop<<<0, 256>>>();
  report("grid x = 0", cudaGetLastError());
  nop<<<1, 2000>>>();
  report("block = 2000 (> 1024)", cudaGetLastError());
  nop<<<1, 32, 200000>>>();
  report("dynamic smem = 200000 B", cudaGetLastError());
}

// ---------------------------------------------------------------------------
static void mode_peekget() {
  printf("== cudaPeekAtLastError vs cudaGetLastError ==\n");
  nop<<<0, 256>>>();                                  // make one error
  report("peek #1", cudaPeekAtLastError());
  report("peek #2 (still set)", cudaPeekAtLastError());
  report("get     (returns+clears)", cudaGetLastError());
  report("get #2  (now clear)", cudaGetLastError());
}

// ---------------------------------------------------------------------------
static void mode_async() {
  printf("== execution fault is ASYNCHRONOUS ==\n");
  int *p = nullptr;
  deref_null<<<1, 1>>>(p);                            // launch "succeeds"
  report("right after <<<>>>", cudaGetLastError());   // usually success!
  report("cudaDeviceSynchronize #1", cudaDeviceSynchronize());
  report("cudaDeviceSynchronize #2", cudaDeviceSynchronize());
  report("cudaGetLastError", cudaGetLastError());
  report("cudaDeviceSynchronize #3", cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
static void mode_streams() {
  printf("== legacy default stream vs stream flags ==\n");
  int *dflag = nullptr, *dout = nullptr;
  cudaMalloc(&dflag, sizeof(int));
  cudaMalloc(&dout, sizeof(int));
  // ~0.1 s at ~2 GHz (and still < ~1 s at an idle low clock): far longer than
  // the few microseconds a concurrent set_flag needs to land, yet short enough
  // that the serialized ("blocking") case just times out and moves on.
  const long long timeout_cycles = 200000000LL;

  // (a) "default" (blocking) stream: it must wait for the prior legacy-default
  //     work (the spin), so the flag can only be set AFTER the spin gives up.
  cudaMemset(dflag, 0, sizeof(int));
  spin_wait<<<1, 1>>>(dflag, dout, timeout_cycles);   // legacy default stream
  cudaStream_t sb;
  cudaStreamCreateWithFlags(&sb, cudaStreamDefault);  // "blocking"
  set_flag<<<1, 1, 0, sb>>>(dflag);
  cudaDeviceSynchronize();
  int hb = -1;
  cudaMemcpy(&hb, dout, sizeof(int), cudaMemcpyDeviceToHost);
  printf("  blocking stream    : spin observed flag = %d  (0 => serialized by implicit sync)\n", hb);
  cudaStreamDestroy(sb);

  // (b) non-blocking stream: no implicit dependency -> runs concurrently.
  cudaMemset(dflag, 0, sizeof(int));
  spin_wait<<<1, 1>>>(dflag, dout, timeout_cycles);   // legacy default stream
  cudaStream_t snb;
  cudaStreamCreateWithFlags(&snb, cudaStreamNonBlocking);
  set_flag<<<1, 1, 0, snb>>>(dflag);
  cudaDeviceSynchronize();
  int hn = -1;
  cudaMemcpy(&hn, dout, sizeof(int), cudaMemcpyDeviceToHost);
  printf("  non-blocking stream: spin observed flag = %d  (1 => concurrent)\n", hn);
  cudaStreamDestroy(snb);

  cudaFree(dflag);
  cudaFree(dout);
}

// ---------------------------------------------------------------------------
static void mode_props() {
  cudaDeviceProp p;
  memset(&p, 0, sizeof(p));
  cudaGetDeviceProperties(&p, 0);
  printf("== launch-related device limits (device 0) ==\n");
  printf("  name                          : %s\n", p.name);
  printf("  compute capability            : %d.%d\n", p.major, p.minor);
  printf("  multiProcessorCount           : %d\n", p.multiProcessorCount);
  printf("  warpSize                      : %d\n", p.warpSize);
  printf("  maxThreadsPerBlock            : %d\n", p.maxThreadsPerBlock);
  printf("  maxThreadsPerMultiProcessor   : %d\n", p.maxThreadsPerMultiProcessor);
  printf("  maxGridSize                   : [%d, %d, %d]\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
  printf("  maxBlockDim                   : [%d, %d, %d]\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
  printf("  sharedMemPerBlock             : %zu\n", (size_t)p.sharedMemPerBlock);
  printf("  sharedMemPerBlockOptin        : %zu\n", (size_t)p.sharedMemPerBlockOptin);
  printf("  regsPerBlock                  : %d\n", p.regsPerBlock);
  printf("  concurrentKernels             : %d\n", p.concurrentKernels);
  printf("  asyncEngineCount              : %d\n", p.asyncEngineCount);
  printf("  canMapHostMemory              : %d\n", p.canMapHostMemory);
  printf("  cooperativeLaunch             : %d\n", p.cooperativeLaunch);
  printf("  streamPrioritiesSupported     : %d\n", p.streamPrioritiesSupported);
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "props";
  if (!strcmp(mode, "config"))       mode_config();
  else if (!strcmp(mode, "peekget")) mode_peekget();
  else if (!strcmp(mode, "async"))   mode_async();
  else if (!strcmp(mode, "streams")) mode_streams();
  else if (!strcmp(mode, "props"))   mode_props();
  else { printf("usage: %s {config|peekget|async|streams|props}\n", argv[0]); return 2; }
  return 0;
}
