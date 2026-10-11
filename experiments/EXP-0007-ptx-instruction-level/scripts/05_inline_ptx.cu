// D07 / 指令级-PTX: inline PTX 操作数约束 + memory clobber
// sm_89 / CUDA 12.8
#include <cuda_runtime.h>
#include <cstdio>

// 用 inline PTX 读 %laneid、%clock，演示约束
__global__ void probe(unsigned* out) {
    unsigned lane;
    unsigned long long clk;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    asm("mov.u64 %0, %%clock64;" : "=l"(clk));
    out[threadIdx.x] = lane + (unsigned)(clk & 1ull);   // clk 高位截断，仅演示
}

// 用约束 "=f" 做浮点、"+r" 做读改写
__global__ void addone(float* x) {
    float v = x[0];
    asm("add.f32 %0, %1, %2;" : "=f"(v) : "f"(v), "f"(1.0f));
    x[0] = v;
}

// 反面示范：自造"屏障"，却漏了 "memory" clobber。
// bar.sync 在硬件上确实是 CTA 屏障，但没有 "memory" 时编译器不知道这条内联汇编会碰内存，
// 于是可以把它周围的访存重排、合并甚至删除——屏障语义被编译器整个绕过。
// 正确写法：asm volatile("bar.sync 0;" ::: "memory")，或干脆用 __syncthreads()。
// 详见 docs/research/07-ptx-instruction-level.md。
__global__ void fake_fence(int* flag, int* data, int* out) {
    *data = 7;
    asm volatile("bar.sync 0;" ::: );   // 缺 "memory" clobber
    *out = *flag;
}

int main() {
    unsigned* o;
    float* x;
    int *f, *d, *out;
    cudaMallocManaged(&o, 32 * sizeof(unsigned));
    cudaMallocManaged(&x, sizeof(float));
    cudaMallocManaged(&f, sizeof(int));
    cudaMallocManaged(&d, sizeof(int));
    cudaMallocManaged(&out, sizeof(int));
    *x = 1.0f;
    probe<<<1, 32>>>(o);
    addone<<<1, 1>>>(x);
    fake_fence<<<1, 1>>>(f, d, out);
    cudaDeviceSynchronize();
    printf("cudaGetLastError=%s x=%f d=%d\n",
           cudaGetErrorString(cudaGetLastError()), *x, *d);
    return 0;
}
