// EXP-0004 补充：memory order × thread scope 矩阵
// 目的：给 docs/research/04 的「order 标尺」配上 SASS 证据——
//       每种 (order, scope) 组合下，"写数据 + 发布/获取 flag" 编译出什么机器码。
// 说明：
//   - 每个 (order, scope) 生成一对 kernel：prod_*/cons_*，动作都是 "data=42; 发布 flag" / "等 flag; 读 data"。
//   - order 只在合法用法上出现：store 只用 relaxed/release/seq_cst；load 只用 relaxed/acquire/seq_cst；
//     acq_rel 属于 RMW，生产者用 fetch_add(1, acq_rel) 表达。
//   - 运行期只做 smoke（kernel 边界会同步，语义诚实性依赖 SASS 而非运行数字）；
//     matrix 模式打印逐组合 "ran / out" 结果。
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cstdio>

using A_dev = cuda::atomic<int, cuda::thread_scope_device>;
using A_sys = cuda::atomic<int, cuda::thread_scope_system>;

// ---- scope = device --------------------------------------------------------
__global__ void prod_dev_relaxed (int* d, A_dev* f)          { *d = 42; f->store(1, cuda::memory_order_relaxed); }
__global__ void cons_dev_relaxed (const int* d, A_dev* f, int* o) { while (f->load(cuda::memory_order_relaxed) != 1) {} *o = *d; }

__global__ void prod_dev_release (int* d, A_dev* f)          { *d = 42; f->store(1, cuda::memory_order_release); }
__global__ void cons_dev_release (const int* d, A_dev* f, int* o) { while (f->load(cuda::memory_order_relaxed) != 1) {} *o = *d; }

__global__ void prod_dev_acquire (int* d, A_dev* f)          { *d = 42; f->store(1, cuda::memory_order_relaxed); }
__global__ void cons_dev_acquire (const int* d, A_dev* f, int* o) { while (f->load(cuda::memory_order_acquire) != 1) {} *o = *d; }

__global__ void prod_dev_acq_rel (int* d, A_dev* f)          { *d = 42; f->fetch_add(1, cuda::memory_order_acq_rel); }
__global__ void cons_dev_acq_rel (const int* d, A_dev* f, int* o) { while (f->load(cuda::memory_order_acquire) != 1) {} *o = *d; }

__global__ void prod_dev_seq_cst (int* d, A_dev* f)          { *d = 42; f->store(1, cuda::memory_order_seq_cst); }
__global__ void cons_dev_seq_cst (const int* d, A_dev* f, int* o) { while (f->load(cuda::memory_order_seq_cst) != 1) {} *o = *d; }

// ---- scope = system --------------------------------------------------------
__global__ void prod_sys_relaxed (int* d, A_sys* f)          { *d = 42; f->store(1, cuda::memory_order_relaxed); }
__global__ void cons_sys_relaxed (const int* d, A_sys* f, int* o) { while (f->load(cuda::memory_order_relaxed) != 1) {} *o = *d; }

__global__ void prod_sys_release (int* d, A_sys* f)          { *d = 42; f->store(1, cuda::memory_order_release); }
__global__ void cons_sys_release (const int* d, A_sys* f, int* o) { while (f->load(cuda::memory_order_relaxed) != 1) {} *o = *d; }

__global__ void prod_sys_acquire (int* d, A_sys* f)          { *d = 42; f->store(1, cuda::memory_order_relaxed); }
__global__ void cons_sys_acquire (const int* d, A_sys* f, int* o) { while (f->load(cuda::memory_order_acquire) != 1) {} *o = *d; }

__global__ void prod_sys_acq_rel (int* d, A_sys* f)          { *d = 42; f->fetch_add(1, cuda::memory_order_acq_rel); }
__global__ void cons_sys_acq_rel (const int* d, A_sys* f, int* o) { while (f->load(cuda::memory_order_acquire) != 1) {} *o = *d; }

__global__ void prod_sys_seq_cst (int* d, A_sys* f)          { *d = 42; f->store(1, cuda::memory_order_seq_cst); }
__global__ void cons_sys_seq_cst (const int* d, A_sys* f, int* o) { while (f->load(cuda::memory_order_seq_cst) != 1) {} *o = *d; }

int main() {
    A_dev* f_dev; A_sys* f_sys;
    int *data, *out;
    cudaMallocManaged(&f_dev, sizeof(A_dev));
    cudaMallocManaged(&f_sys, sizeof(A_sys));
    cudaMallocManaged(&data, 4);
    cudaMallocManaged(&out, 4);
    // smoke：每个组合各跑一次（kernel 边界同步，只验证能跑通、无崩溃）
    int rc = 0;
    #define RUN_DEV(TAG) \
        do { \
            f_dev->store(0, cuda::memory_order_relaxed); *data = 0; *out = 0; \
            prod_dev_##TAG<<<1,1>>>(data, f_dev); \
            cons_dev_##TAG<<<1,1>>>(data, f_dev, out); \
            cudaError_t e = cudaDeviceSynchronize(); \
            printf("dev." #TAG " : err=%d out=%d\n", (int)e, *out); \
            if (e != cudaSuccess || *out != 42) rc = 1; \
        } while (0)
    #define RUN_SYS(TAG) \
        do { \
            f_sys->store(0, cuda::memory_order_relaxed); *data = 0; *out = 0; \
            prod_sys_##TAG<<<1,1>>>(data, f_sys); \
            cons_sys_##TAG<<<1,1>>>(data, f_sys, out); \
            cudaError_t e = cudaDeviceSynchronize(); \
            printf("sys." #TAG " : err=%d out=%d\n", (int)e, *out); \
            if (e != cudaSuccess || *out != 42) rc = 1; \
        } while (0)

    RUN_DEV(relaxed); RUN_DEV(release); RUN_DEV(acquire); RUN_DEV(acq_rel); RUN_DEV(seq_cst);
    RUN_SYS(relaxed); RUN_SYS(release); RUN_SYS(acquire); RUN_SYS(acq_rel); RUN_SYS(seq_cst);
    printf("smoke rc=%d\n", rc);
    return rc;
}
