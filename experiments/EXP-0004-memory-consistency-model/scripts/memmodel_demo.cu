// memmodel_demo.cu — D04 (内存一致性模型) 最小探针，目标 sm_89 / CUDA 12.8
//
// 目的：把「弱内存模型 + 作用域 + 顺序」的承诺差距变成可观察的产物：
//   1) volatile != atomic          —— 会丢更新的反例
//   2) 生产者三种写法              —— 供 cuobjdump -sass 对照（有无 MEMBAR / 强弱访存）
//   3) 跨 block 消息传递           —— naive（仅 volatile）vs release/acquire
//
// 编译：先 vcvars64.bat，再 nvcc -O3 -lineinfo -arch=sm_89
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
  printf("[CUDA ERROR] %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(_e)); \
  exit(2); } } while (0)

// ------------------------------------------------------------------
// 1. volatile 不是原子：读-改-写 会丢更新
// ------------------------------------------------------------------
__global__ void k_volatile_inc(volatile int* c, int iters) {
    for (int i = 0; i < iters; ++i) { int v = *c; *c = v + 1; }
}
__global__ void k_atomic_inc(int* c, int iters) {
    for (int i = 0; i < iters; ++i) atomicAdd(c, 1);
}

// ------------------------------------------------------------------
// 2. 生产者写法对照（SASS/PTX 用）：写 data 再写 flag
// ------------------------------------------------------------------
__global__ void k_prod_naive(int* data, int* flag) {         // 无任何同步
    *data = 42;
    *flag = 1;
}
__global__ void k_prod_fence_block(int* data, volatile int* flag) {
    *data = 42;
    __threadfence_block();                                    // 只管 block 作用域
    *flag = 1;
}
__global__ void k_prod_fence_device(int* data, volatile int* flag) {
    *data = 42;
    __threadfence();                                          // device 作用域 seq_cst fence
    *flag = 1;
}
__global__ void k_prod_fence_system(int* data, volatile int* flag) {
    *data = 42;
    __threadfence_system();                                   // system 作用域 seq_cst fence
    *flag = 1;
}
__global__ void k_prod_atomic_relaxed(int* data, int* flag) { // 直觉：原子就是同步？——默认 relaxed
    *data = 42;
    atomicAdd(flag, 1);
}
__global__ void k_prod_release(int* data, int* flag) {        // release store（libcu++ builtin）
    *data = 42;
    __nv_atomic_store_n(flag, 1, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE);
}

// ------------------------------------------------------------------
// 3. 跨 block 消息传递：block0 生产，block1 消费
//    flag 两个版本都用 volatile 自旋，避免自旋被编译器提升；
//    差别只在「data 写与 flag 写之间有没有 release 语义」。
//
//    注意：本 kernel 用 <<<2,32>>>，block0 生产、block1 消费。CUDA 不保证
//    两个 block 同时驻留，所以这里的自旋依赖「另一个 block 恰好在跑」。消费者
//    的自旋有上界 spin_max（教学保护，超时会计入 timeouts 而不是挂住）——但这
//    只是不让实验挂死，不等于算法安全：跨 block 自旋要么保证驻留（如
//    cooperative launch + grid.sync），要么就别假设另一端会并行推进。
// ------------------------------------------------------------------
__global__ void k_msg_naive(volatile int* data, volatile int* flag, int* out, int spin_max) {
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) { *data = 12345; *flag = 1; }   // 顺序无保证
    } else {
        if (threadIdx.x == 0) {
            int s = 0;
            while (*flag == 0 && s < spin_max) ++s;
            out[0] = *data;                                   // 可能读到旧值
            out[1] = (s < spin_max) ? 1 : 0;                  // 1 = 收到了 flag
        }
    }
}
__global__ void k_msg_acqrel(volatile int* data, int* flag, int* out, int spin_max) {
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            *data = 12345;
            __nv_atomic_store_n(flag, 1, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE);
        }
    } else {
        if (threadIdx.x == 0) {
            int s = 0;
            while (__nv_atomic_load_n(flag, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE) == 0
                   && s < spin_max) ++s;
            out[0] = *data;
            out[1] = (s < spin_max) ? 1 : 0;
        }
    }
}

// ------------------------------------------------------------------
// host
// ------------------------------------------------------------------
static void mode_inc(const char* which, int grid, int block, int iters) {
    int* d; CK(cudaMalloc(&d, sizeof(int))); CK(cudaMemset(d, 0, sizeof(int)));
    if (which[0] == 'v') k_volatile_inc<<<grid, block>>>(d, iters);
    else                 k_atomic_inc<<<grid, block>>>(d, iters);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int h = 0; CK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
    long long expected = (long long)grid * block * iters;
    printf("mode=%s grid=%d block=%d iters=%d expected=%lld observed=%d lost=%lld\n",
           which, grid, block, iters, expected, h, expected - (long long)h);
    CK(cudaFree(d));
}

static void mode_producer_once(const char* name, int variant) {
    int *data, *flag, *hdata, *hflag;
    CK(cudaMalloc(&data, sizeof(int))); CK(cudaMalloc(&flag, sizeof(int)));
    CK(cudaMemset(data, 0, sizeof(int))); CK(cudaMemset(flag, 0, sizeof(int)));
    switch (variant) {
        case 0: k_prod_naive<<<1, 1>>>(data, flag); break;
        case 1: k_prod_fence_block<<<1, 1>>>(data, (volatile int*)flag); break;
        case 2: k_prod_fence_device<<<1, 1>>>(data, (volatile int*)flag); break;
        case 3: k_prod_fence_system<<<1, 1>>>(data, (volatile int*)flag); break;
        case 4: k_prod_atomic_relaxed<<<1, 1>>>(data, flag); break;
        case 5: k_prod_release<<<1, 1>>>(data, flag); break;
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMallocHost((void**)&hdata, sizeof(int)));
    CK(cudaMallocHost((void**)&hflag, sizeof(int)));
    CK(cudaMemcpy(hdata, data, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hflag, flag, sizeof(int), cudaMemcpyDeviceToHost));
    printf("producer=%-18s data=%d flag=%d\n", name, *hdata, *hflag);
    CK(cudaFreeHost(hdata)); CK(cudaFreeHost(hflag));
    CK(cudaFree(data)); CK(cudaFree(flag));
}

static void mode_msg(const char* name, int acqrel, int iters, int spin_max) {
    volatile int* data; int* flag; int* out; int h[2];
    CK(cudaMalloc((void**)&data, sizeof(int))); CK(cudaMalloc(&flag, sizeof(int)));
    CK(cudaMalloc(&out, 2 * sizeof(int)));
    int violations = 0, received = 0, timeouts = 0;
    for (int i = 0; i < iters; ++i) {
        CK(cudaMemset((void*)data, 0, sizeof(int)));
        CK(cudaMemset(flag, 0, sizeof(int)));
        CK(cudaMemset(out, 0, 2 * sizeof(int)));
        if (acqrel) k_msg_acqrel<<<2, 32>>>(data, flag, out, spin_max);
        else        k_msg_naive  <<<2, 32>>>(data, flag, out, spin_max);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h, out, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        if (h[1]) { received++; if (h[0] != 12345) violations++; }
        else timeouts++;
    }
    printf("mode=%s iters=%d received_flag=%d violations(stale data)=%d timeouts=%d\n",
           name, iters, received, violations, timeouts);
    CK(cudaFree((void*)data)); CK(cudaFree(flag)); CK(cudaFree(out));
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "help";
    if (!strcmp(mode, "volatile-nonatomic")) { mode_inc("volatile", 64, 256, argc > 2 ? atoi(argv[2]) : 4000); }
    else if (!strcmp(mode, "atomic"))        { mode_inc("atomic",   64, 256, argc > 2 ? atoi(argv[2]) : 4000); }
    else if (!strcmp(mode, "producers")) {
        mode_producer_once("naive",          0);
        mode_producer_once("fence_block",    1);
        mode_producer_once("fence_device",   2);
        mode_producer_once("fence_system",   3);
        mode_producer_once("atomic_relaxed", 4);
        mode_producer_once("release",        5);
    }
    else if (!strcmp(mode, "msg-naive"))  { mode_msg("msg-naive",  0, argc > 2 ? atoi(argv[2]) : 2000, 20000000); }
    else if (!strcmp(mode, "msg-acqrel")) { mode_msg("msg-acqrel", 1, argc > 2 ? atoi(argv[2]) : 2000, 20000000); }
    else {
        printf("usage: %s <volatile-nonatomic|atomic|producers|msg-naive|msg-acqrel> [iters]\n", argv[0]);
        return 1;
    }
    return 0;
}
