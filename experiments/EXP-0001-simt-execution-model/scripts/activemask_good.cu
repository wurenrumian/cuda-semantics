// EXP-0001 补充：__activemask() 的正当用法（未决 ⑤）
// 两个合法模式（相对于反例 A 的错误用法）：
//   discovery: 发散分支内，activemask 快照 + __syncwarp(同 mask) 循环到"集合稳定"，
//              得到分支内公认的参与集合；通信/归约只用这个集合内的 lane（源 lane 一定在 mask 里）。
//   contrast:  同一分支内直接 __ballot_sync(0xffffffff) —— mask 含未参与 lane = UB（对照，仅演示会怎样，不打印值靠检查）。
// 输出（模式 discovery）：
//   每个偶数 lane：收敛后的 mask、mask 内 lane 数、从 (lane+2)&31 shuffle 得到的值（期望 (lane+2)*10）
#include <cstdio>
#include <cuda_runtime.h>

// pred=false 的 lane 不能充当有效源时给出占位值；保持"全员执行同一条 intrinsic"
__device__ inline int premasked(int v, int pred) { return pred ? v : 0; }

__global__ void k_discovery(unsigned* mask_out, int* val_out, int* nerr) {
    int lane = threadIdx.x & 31;
    int v = lane * 10;
    if (lane & 1) return;                        // 只有偶数 lane 进入（对应反例 A 的发散形状）
    unsigned m0, m1;
    do {                                          // Discovery：快照 -> 分支屏障 -> 再快照，直到稳定
        m0 = __activemask();
        __syncwarp(m0);
        m1 = __activemask();
    } while (m0 != m1);
    int src = (lane + 2) & 31;                    // 下一偶数 lane（wrap 30->0，保持子集内）
    int x = __shfl_sync(m1, v, src);
    atomicAdd(nerr, (x != ((src) * 10)) ? 1 : 0);
    if (lane == 0) *mask_out = m1;
    val_out[lane] = x;
}

__global__ void k_baseline(int* val_out) {        // 收敛对照：全 warp 参与（不依赖 activemask）
    int lane = threadIdx.x & 31;
    int v = lane * 10;
    int pred = (lane & 1) == 0;
    // 所有 lane 都执行同一条 intrinsic（mask=全员），谓词只决定"记不记"
    unsigned m = __ballot_sync(0xffffffff, pred);  // 决策用：谁记
    int src = (lane + 2) & 31;                     // 即使偶 lane 的 src 也是偶数（保持子集内）
    int x = __shfl_sync(0xffffffff, premasked(v, pred), src);
    if (pred) val_out[lane] = x;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "discovery";
    unsigned* mask; int *val, *nerr;
    cudaMallocManaged(&mask, 4); cudaMallocManaged(&val, 32*4); cudaMallocManaged(&nerr, 4);
    *nerr = 0; *mask = 0;
    if (mode[0] == 'd') {
        k_discovery<<<1, 32>>>(mask, val, nerr);
        cudaDeviceSynchronize();
        unsigned popc = 0; for (unsigned t = *mask; t; t &= (t-1)) popc++;
        printf("discovery: mask=0x%08x  popc=%u  mismatches=%d\n", *mask, popc, *nerr);
        for (int l = 0; l < 32; l += 2) printf("  %2d : got %3d want %3d %s\n", l, val[l], ((l+2)&31)*10, val[l]==((l+2)&31)*10?"":"<-- WRONG");
    } else {
        k_baseline<<<1, 32>>>(val);
        cudaDeviceSynchronize();
        int bad = 0;
        for (int l = 0; l < 32; l += 2) if (val[l] != ((l+2)&31)*10) bad++;
        printf("baseline: mismatches=%d (expect 0)\n", bad);
    }
    return 0;
}
