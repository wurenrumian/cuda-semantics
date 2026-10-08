// D07 / 指令级-PTX: 用 inline PTX 直接写 cache operator，排除 intrinsic 包装的干扰
// 目的: 看清 ld.global.<cop> 与 默认 ld.global、volatile 在 SASS 里的真实映射
// sm_89 / CUDA 12.8
#include <cuda_runtime.h>
#include <cstdio>

__global__ void cop_probe(const int* p, int* out) {
    int a, b, c, d, e, f, g;
    asm volatile("ld.global.b32    %0,[%1];" : "=r"(a) : "l"(p) : "memory");
    asm volatile("ld.global.ca.b32 %0,[%1];" : "=r"(b) : "l"(p) : "memory");
    asm volatile("ld.global.cg.b32 %0,[%1];" : "=r"(c) : "l"(p) : "memory");
    asm volatile("ld.global.cs.b32 %0,[%1];" : "=r"(d) : "l"(p) : "memory");
    asm volatile("ld.global.lu.b32 %0,[%1];" : "=r"(e) : "l"(p) : "memory");
    asm volatile("ld.global.cv.b32 %0,[%1];" : "=r"(f) : "l"(p) : "memory");
    asm volatile("ld.volatile.global.b32 %0,[%1];" : "=r"(g) : "l"(p) : "memory");
    out[0] = a + b + c + d + e + f + g;
}

__global__ void cop_store(int* p, int v) {
    asm volatile("st.global.wb.b32 [%0],%1;" :: "l"(p), "r"(v) : "memory");
    asm volatile("st.global.cg.b32 [%0],%1;" :: "l"(p), "r"(v) : "memory");
    asm volatile("st.global.cs.b32 [%0],%1;" :: "l"(p), "r"(v) : "memory");
    asm volatile("st.global.wt.b32 [%0],%1;" :: "l"(p), "r"(v) : "memory");
}

int main() {
    int *p, *out;
    cudaMallocManaged(&p, sizeof(int));
    cudaMallocManaged(&out, sizeof(int));
    *p = 1;
    cop_probe<<<1, 1>>>(p, out);
    cop_store<<<1, 1>>>(p, 9);
    cudaDeviceSynchronize();
    printf("cudaGetLastError=%s out=%d\n",
           cudaGetErrorString(cudaGetLastError()), *out);
    return 0;
}
