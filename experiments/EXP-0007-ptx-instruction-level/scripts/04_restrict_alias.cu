// D07 反例: 违背 const __restrict__ 承诺 → 编译器复用早先的 load，读到"旧值"
// 期望(想当然): 写 5 之后再读，应得 5
// 实际: 得到 1（第二次 p[0] 被 CSE 掉，没有重新加载）
// sm_89 / CUDA 12.8
#include <cuda_runtime.h>
#include <cstdio>

// p 被声明为 const __restrict__：向编译器承诺"p 不与其他指针别名"
__global__ void alias_break(float* q, const float* __restrict__ p, float* out) {
    float before = p[0];      // 加载 #1
    *q = 5.0f;                // 通过 q 写 —— 实际上 q 与 p 指向同一块内存
    float after = p[0];       // 加载 #2：编译器认为 p 不受 *q 影响 → 复用 before
    out[0] = before;
    out[1] = after;
}

// 对照：p 不用 restrict，编译器必须重新加载
__global__ void alias_ok(float* q, const float* p, float* out) {
    float before = p[0];
    *q = 5.0f;
    float after = p[0];
    out[0] = before;
    out[1] = after;
}

int main() {
    float* buf;
    float* out;
    cudaMallocManaged(&buf, 4 * sizeof(float));
    cudaMallocManaged(&out, 4 * sizeof(float));
    buf[0] = 1.0f;

    alias_break<<<1, 1>>>(buf, buf, out);   // q == p == buf
    cudaDeviceSynchronize();
    printf("[restrict] before=%f after=%f (buf now=%f)\n", out[0], out[1], buf[0]);

    buf[0] = 1.0f;
    alias_ok<<<1, 1>>>(buf, buf, out);
    cudaDeviceSynchronize();
    printf("[plain   ] before=%f after=%f (buf now=%f)\n", out[0], out[1], buf[0]);
    printf("cudaGetLastError=%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
