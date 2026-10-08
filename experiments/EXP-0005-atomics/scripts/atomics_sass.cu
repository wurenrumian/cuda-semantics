// D05 atomics lab -- instruction-level kernels (PTX / SASS inspection only).
// Target: sm_89 (Ada, RTX 4060 Laptop) / CUDA 12.8.
// These kernels are compiled to PTX and to a cubin; they are never launched.
// Purpose: see what the toolchain actually emits for "atomic" vs "fence" vs
// libcu++ ordered atomics, so the atomicity / ordering distinction is grounded
// in generated code, not folklore.
#include <cuda_runtime.h>
#include <cuda/atomic>

__device__ int    g_c;        // int counter (global)
__device__ int    g_ready;    // flag
__device__ int    g_data[8];
__device__ float  g_f;
__device__ double g_d;

// ---- default device intrinsics: atomic, but relaxed ordering --------------
__global__ void k_atom_add(int* out, int v) { out[threadIdx.x] = atomicAdd(&g_c, v); }
__global__ void k_atom_exch(int* out, int v) { out[threadIdx.x] = atomicExch(&g_c, v); }
__global__ void k_atom_cas(int* out, int a, int b) { out[threadIdx.x] = atomicCAS(&g_c, a, b); }

// return value unused -> compiler may lower to a "reduction" (no return value)
__global__ void k_red_add(int* out, int v) { atomicAdd(&g_c, v); out[threadIdx.x] = 0; }

// shared-memory atomic (parsed in the SM, not L2)
__global__ void k_sh_atom_add(int* out, int v) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  int o = atomicAdd(&s, v);
  __syncthreads();
  out[threadIdx.x] = o;
}

// float / double atomics (sm_20 / sm_60 in the headers)
__global__ void k_atom_add_f32(float* out, float v) { out[threadIdx.x] = atomicAdd(&g_f, v); }
__global__ void k_atom_add_f64(double* out, double v) { out[threadIdx.x] = atomicAdd(&g_d, v); }

// scope variants: block (cta) / device (default) / system
__global__ void k_atom_add_block(int* out, int v) { out[threadIdx.x] = atomicAdd_block(&g_c, v); }
__global__ void k_atom_add_system(int* out, int v) { out[threadIdx.x] = atomicAdd_system(&g_c, v); }

// ---- fences: what "__threadfence" actually costs --------------------------
__global__ void k_fence_gpu(int* out)   { __threadfence();         out[threadIdx.x] = g_ready; }
__global__ void k_fence_block(int* out) { __threadfence_block();   out[threadIdx.x] = g_ready; }
__global__ void k_fence_sys(int* out)   { __threadfence_system();  out[threadIdx.x] = g_ready; }

// ---- libcu++ cuda::atomic_ref: order is explicit --------------------------
using aref = cuda::atomic_ref<int, cuda::thread_scope_device>;

__global__ void k_lib_relaxed_add(int* out, int v) { aref a(g_c); out[threadIdx.x] = a.fetch_add(v, cuda::memory_order_relaxed); }
__global__ void k_lib_seqcst_add(int* out, int v) { aref a(g_c); out[threadIdx.x] = a.fetch_add(v, cuda::memory_order_seq_cst); }
__global__ void k_lib_release_store(int* out, int v) { aref a(g_ready); a.store(v, cuda::memory_order_release); out[threadIdx.x] = 0; }
__global__ void k_lib_acquire_load(int* out) { aref a(g_ready); out[threadIdx.x] = a.load(cuda::memory_order_acquire); }
