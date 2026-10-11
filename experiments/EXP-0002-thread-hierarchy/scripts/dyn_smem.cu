// EXP-0002 补充：动态 shared memory 与 __launch_bounds__ 的联合占用效应
// 问题（对应研究稿 02「动态 shared memory」小节）：
//   1) 静态和动态 shared memory 是否在同一池子里计入 occupancy？（理论：是）
//   2) 给同一个 kernel 加 __launch_bounds__(128, 8)，在有/无动态 shared 下分别落出什么
//      （寄存器、spill、blocks/SM）？
//   3) 只传 sharedMemPerBlock（block 静态上限）而 kernel 实际用 dynamic 时，cudaOccupancy API
//      是否需要显式传 dynamicBytes？（模式 underreport 演示：漏传会把 blocks/SM 高估）
// 模式：dyn（默认）—— 三组实测数字；lb —— launch_bounds 变体。
#include <cstdio>
#include <cuda_runtime.h>

__shared__ int s_static[4096];                 // 16 KB 静态

__global__ void k_static16k(const int* in, int* out) {
    s_static[threadIdx.x] = in[threadIdx.x];   // 触碰静态 shared
    __syncthreads();
    out[threadIdx.x] = s_static[threadIdx.x];
}

__global__ void k_dyn16k(const int* in, int* out) {   // 无静态 shared，16 KB 动态
    extern __shared__ int s_dyn[];
    s_dyn[threadIdx.x] = in[threadIdx.x];
    __syncthreads();
    out[threadIdx.x] = s_dyn[threadIdx.x];
}

template<int MINB>
__global__ void __launch_bounds__(128, MINB) k_dyn16k_lb(const int* in, int* out) {
    extern __shared__ int s_dyn[];
    s_dyn[threadIdx.x] = in[threadIdx.x];
    __syncthreads();
    out[threadIdx.x] = s_dyn[threadIdx.x];
}

int occ_of(const void* k, int block, size_t dyn) {
    int n = -1;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, block, dyn);
    return (e == cudaSuccess) ? n : -2;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "dyn";
    void* in; void* out;
    cudaMallocManaged(&in, 128*4); cudaMallocManaged(&out, 128*4);
    k_static16k<<<1, 128>>>((int*)in, (int*)out);   // 触发模块加载
    cudaDeviceSynchronize();

    printf("== occupancy: static 16KB vs dynamic 16KB (block=128) ==\n");
    printf("k_static16k                  blocks/SM = %d\n", occ_of((void*)k_static16k, 128, 0));
    // 漏报动态尺寸（只传 0） vs 如实传 16KB：
    printf("k_dyn16k   dyn=0 (漏报:underreport)      blocks/SM = %d\n", occ_of((void*)k_dyn16k, 128, 0));
    printf("k_dyn16k   dyn=16KB (如实)   blocks/SM = %d\n", occ_of((void*)k_dyn16k, 128, 16384));
    puts("    （本机 sharedMemPerMultiprocessor 较小时漏报会显著高估 blocks/SM） ");

    printf("== launch_bounds(128,8) x dynamic smem ==\n");
    printf("k_dyn16k_lb<8> dyn=16KB      blocks/SM = %d", occ_of((void*)k_dyn16k_lb<8>, 128, 16384));
    puts("   （请求 8；下界不封顶，smem 是硬上限） ");
    printf("k_dyn16k_lb<8> dyn=0         blocks/SM = %d\n", occ_of((void*)k_dyn16k_lb<8>, 128, 0));
    return 0;
}
