// D07 / 指令级-PTX: 弱访存 vs 强访存(acquire/release) vs volatile
// sm_89 / CUDA 12.8
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdio>

// plain (weak) load / store
__global__ void weak_k(const int* __restrict__ p, int* __restrict__ out) {
    int v = *p;
    *out = v;
}

// strong load: acquire at gpu scope, by inline PTX
__global__ void strongld_k(const int* p, int* out) {
    int v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(v) : "l"(p) : "memory");
    *out = v;
}

// strong store: release at gpu scope, by inline PTX
__global__ void strongst_k(int* p, int v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                 :: "l"(p), "r"(v) : "memory");
}

// volatile-qualified load
__global__ void volat_k(const int* p, int* out) {
    *out = *(volatile const int*)p;
}

// libcu++ atomic_ref: relaxed vs acquire/release vs seq_cst
__global__ void acqrel_k(const int* p, int* out) {
    cuda::atomic_ref<int, cuda::thread_scope_device> a(*const_cast<int*>(p));
    int v = a.load(cuda::memory_order_acquire);
    cuda::atomic_ref<int, cuda::thread_scope_device> o(*out);
    o.store(v, cuda::memory_order_release);
}

__global__ void seqcst_k(const int* p, int* out) {
    cuda::atomic_ref<int, cuda::thread_scope_system> a(*const_cast<int*>(p));
    *out = a.load(cuda::memory_order_seq_cst);
}

int main() {
    int *p, *out;
    cudaMallocManaged(&p, sizeof(int));
    cudaMallocManaged(&out, sizeof(int));
    *p = 42;
    weak_k<<<1, 1>>>(p, out);
    strongld_k<<<1, 1>>>(p, out);
    strongst_k<<<1, 1>>>(p, 7);
    volat_k<<<1, 1>>>(p, out);
    acqrel_k<<<1, 1>>>(p, out);
    seqcst_k<<<1, 1>>>(p, out);
    cudaDeviceSynchronize();
    printf("cudaGetLastError=%s out=%d\n",
           cudaGetErrorString(cudaGetLastError()), *out);
    return 0;
}
