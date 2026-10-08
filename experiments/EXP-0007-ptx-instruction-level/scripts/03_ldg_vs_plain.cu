// D07 / 指令级-PTX: __ldg / const __restrict__ / 死代码消除 / 循环外提
// sm_89 / CUDA 12.8
#include <cuda_runtime.h>
#include <cstdio>

// __ldg 显式只读路径 vs 普通加载（都是只读输入）
__global__ void via_ldg(const float* p, float* out) {
    out[threadIdx.x] = __ldg(&p[threadIdx.x]);
}
__global__ void via_plain(const float* __restrict__ p, float* __restrict__ out) {
    out[threadIdx.x] = p[threadIdx.x];
}

// 结果未被使用的加载 —— 会被整条删除吗？
__global__ void dead_load(const float* p, float* out) {
    float unused = p[0];      // 只有这个 load，结果没人用
    out[0] = 3.0f;
}

// const __restrict__ 承诺下，循环不变量加载可被外提
__global__ void hoist(const float* __restrict__ p, float* out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += p[0];   // p[0] 循环不变量
    out[0] = acc;
}

int main() {
    float *p, *out;
    cudaMallocManaged(&p, 64 * sizeof(float));
    cudaMallocManaged(&out, 64 * sizeof(float));
    for (int i = 0; i < 64; i++) p[i] = 1.0f;
    via_ldg<<<1, 32>>>(p, out);
    via_plain<<<1, 32>>>(p, out);
    dead_load<<<1, 1>>>(p, out);
    hoist<<<1, 1>>>(p, out, 1000000);
    cudaDeviceSynchronize();
    printf("cudaGetLastError=%s out[0]=%f\n",
           cudaGetErrorString(cudaGetLastError()), out[0]);
    return 0;
}
