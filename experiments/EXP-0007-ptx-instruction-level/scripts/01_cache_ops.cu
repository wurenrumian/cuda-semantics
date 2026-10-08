// D07 / 指令级-PTX: cache operator 语义观察
// sm_89 / CUDA 12.8
#include <cuda_runtime.h>
#include <cstdio>

__global__ void cache_ops(const float* __restrict__ in, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float ca = __ldca(&in[i]);  // cache at all levels
    float cg = __ldcg(&in[i]);  // cache at L2 only (bypass L1)
    float cs = __ldcs(&in[i]);  // streaming, evict-first
    float cv = __ldcv(&in[i]);  // don't cache, fetch again
    float lu = __ldlu(&in[i]);  // last use
    float sum = ca + cg + cs + cv + lu;
    __stwb(&out[i], sum);       // write-back
    __stcg(&out[i], ca);        // cache global (L2)
    __stcs(&out[i], cg);        // streaming
    __stwt(&out[i], cs);        // write-through
}

int main() {
    const int n = 32;
    float *in, *out;
    cudaMallocManaged(&in, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));
    for (int i = 0; i < n; i++) in[i] = 1.0f;
    cache_ops<<<1, n>>>(in, out);
    cudaDeviceSynchronize();
    printf("cudaGetLastError=%s out[0]=%f\n",
           cudaGetErrorString(cudaGetLastError()), out[0]);
    return 0;
}
