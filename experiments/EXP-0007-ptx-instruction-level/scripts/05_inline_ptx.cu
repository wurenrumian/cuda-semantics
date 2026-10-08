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

// 故意不带 "memory" clobber 的"自造屏障"（反面示范，见正文）
__global__ void fake_fence(int* flag, int* data, int* out) {
    *data = 7;
    asm volatile("bar.sync 0;" ::: );   // 缺 memory clobber
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
