// EXP-0008 — 浮点与数值语义 (sm_89 / CUDA 12.8)
//
// 用同一份源码在四组编译选项下构建，观察：
//   1) FMA 收缩：a*b+c 默认被收缩为 fma.rn.f32（--fmad 默认 true）
//   2) FTZ / denormal：--ftz=true 或 -use_fast_math 把 denormal 结果/输入冲零
//   3) fast math：改掉 div / sqrt / sin 的语义（近似指令）
//   4) atomicAdd(float) 的求和不保证可复现（顺序相关）
//
// 用法：
//   fp_demo fp     打印各浮点表达式结果的 bit pattern
//   fp_demo sum N  用 atomicAdd(float) 求和，重复 N 次，打印每次的 bit pattern
//
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x);                                    \
  if (e_ != cudaSuccess) {                                                  \
    printf("CUDA error %s at line %d: %s\n", #x, __LINE__,                \
           cudaGetErrorString(e_));                                         \
    std::exit(1);                                                           \
  } } while (0)

__device__ __forceinline__ unsigned fbits(float f) { return __float_as_uint(f); }

// ---------------------------------------------------------------------------
// 浮点表达式观察 kernel
// ---------------------------------------------------------------------------
__global__ void k_fp(float a, float b, float c, float m,
                     float p, float q, unsigned* keep, int* denz) {
  // (1) FMA 收缩：a*b+c 由编译器决定是否收缩为单条 fma
  const float r_contract = a * b + c;
  const float r_fma      = __fmaf_rn(a, b, c);   // 显式融合
  const float r_mul      = __fmul_rn(a, b);       // 显式先乘（正确舍入）
  const float r_muladd   = __fadd_rn(r_mul, c);   // 显式后加（两步入 IEEE）

  // (2) FTZ / denormal
  const float r_den      = m * m;                 // 1e-20 * 1e-20 -> ~1e-40 (denormal)
  const int   den_iszero = (r_den == 0.0f) ? 1 : 0;
  const float d          = 1e-40f;                // denormal 输入
  const float r_denin    = d * 2.0f;

  // (3) fast math 打到的点
  const float r_div      = p / q;
  const float r_divrn    = __fdiv_rn(p, q);
  const float r_sqrt     = sqrtf(q);
  const float r_sqrtrn   = __fsqrt_rn(q);
  const float r_sin      = sinf(p);
  const float r_nan      = __fdiv_rn(0.0f, 0.0f);

  if (threadIdx.x == 0) {
    printf("contract  a*b+c   : %08x  %.9g\n", fbits(r_contract), r_contract);
    printf("fmaf_rn   a*b+c   : %08x  %.9g\n", fbits(r_fma), r_fma);
    printf("mul_rn+add_rn     : %08x  %.9g\n", fbits(r_muladd), r_muladd);
    printf("mul_rn(a,b)       : %08x  %.9g\n", fbits(r_mul), r_mul);
    printf("den       m*m     : %08x  %.9g  iszero=%d\n",
           fbits(r_den), r_den, den_iszero);
    printf("denin     d*2     : %08x  %.9g\n", fbits(r_denin), r_denin);
    printf("div       p/q     : %08x  %.9g\n", fbits(r_div), r_div);
    printf("div_rn    p/q     : %08x  %.9g\n", fbits(r_divrn), r_divrn);
    printf("sqrtf(q)          : %08x  %.9g\n", fbits(r_sqrt), r_sqrt);
    printf("fsqrt_rn(q)       : %08x  %.9g\n", fbits(r_sqrtrn), r_sqrtrn);
    printf("sinf(p)           : %08x  %.9g\n", fbits(r_sin), r_sin);
    printf("0/0               : %08x  %s\n",
           fbits(r_nan), (r_nan != r_nan) ? "nan" : "NOT-nan");
  }

  // 防止整个计算被优化掉
  keep[threadIdx.x] = fbits(r_contract) ^ fbits(r_fma) ^ fbits(r_mul)
                    ^ fbits(r_den) ^ fbits(r_denin) ^ fbits(r_div)
                    ^ fbits(r_sqrt) ^ fbits(r_sin) ^ fbits(r_nan)
                    ^ (unsigned)den_iszero;
  denz[threadIdx.x] = den_iszero;
}

// ---------------------------------------------------------------------------
// 求和 kernel：每线程先做本地串行求和，再一次 atomicAdd 到全局累加器。
// 线程之间 atomicAdd 的到达顺序由调度决定，不可依赖。
// ---------------------------------------------------------------------------
__global__ void k_sum(const float* v, long n, float* out) {
  float s = 0.0f;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long)gridDim.x * blockDim.x) {
    s += v[i];
  }
  atomicAdd(out, s);
}

// ---------------------------------------------------------------------------
// 逐值探针：对一批 q 计算 1/q 与 sqrt(q)，用于找 fast math 下 bit 不同的输入
// ---------------------------------------------------------------------------
__global__ void k_probe(const float* q, int n, float* divout, float* sqrtout) {
  int i = threadIdx.x;
  if (i < n) { divout[i] = 1.0f / q[i]; sqrtout[i] = sqrtf(q[i]); }
}

// ---------------------------------------------------------------------------
// 原子浮点 denormal 行为：global vs shared（与 -ftz 无关，由文档规定）
// ---------------------------------------------------------------------------
__global__ void k_atomden(float* g, float* s_out) {
  __shared__ float sh;
  if (threadIdx.x == 0) sh = 0.0f;
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(&sh, 1e-40f);  // shared：支持 denormal
  __syncthreads();
  if (threadIdx.x == 0) *s_out = sh;
  if (threadIdx.x == 0) atomicAdd(g, 1e-40f);    // global：永远 FTZ
}

static unsigned fbits_host(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "fp";

  if (std::strcmp(mode, "fp") == 0) {
    // 选值：a = 1 + 2^-12, c = -(1 + 2^-11)
    //   exact a*a + c = 2^-24  (= 0x33800000)
    //   round(round(a*a)+c) = 0  (= 0x00000000)  —— 这就是 FMA 收缩的可见差异
    const float a = 1.0f + 0x1p-12f;
    const float b = 1.0f + 0x1p-12f;
    const float c = -(1.0f + 0x1p-11f);
    const float m = 1e-20f;
    const float p = 1.0f;
    const float q = 3.0f;

    const double dref = (double)a * (double)b + (double)c;
    printf("host double a*b+c : %.17g   (float)%08x\n",
           dref, fbits_host((float)dref));
    printf("host float  a*b+c : %08x\n", fbits_host(a * b + c));

    unsigned* keep; int* denz;
    CK(cudaMalloc(&keep, 32 * sizeof(unsigned)));
    CK(cudaMalloc(&denz, 32 * sizeof(int)));
    printf("--- device ---\n");
    k_fp<<<1, 32>>>(a, b, c, m, p, q, keep, denz);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(keep)); CK(cudaFree(denz));
    return 0;
  }

  if (std::strcmp(mode, "probe") == 0) {
    const float qs[] = {3.0f, 7.0f, 11.0f, 13.0f, 17.0f, 19.0f, 23.0f,
                        29.0f, 31.0f, 37.0f, 1.0f / 3.0f, 1e-3f, 1e3f,
                        1.5f, 2.0f, 2.5f, 6.2831853f, 0.1f, 0.7f, 123.456f};
    const int n = (int)(sizeof(qs) / sizeof(qs[0]));
    float *dq, *dd, *ds;
    CK(cudaMalloc(&dq, n * sizeof(float)));
    CK(cudaMalloc(&dd, n * sizeof(float)));
    CK(cudaMalloc(&ds, n * sizeof(float)));
    CK(cudaMemcpy(dq, qs, n * sizeof(float), cudaMemcpyHostToDevice));
    k_probe<<<1, 32>>>(dq, n, dd, ds);
    CK(cudaDeviceSynchronize());
    float* hd = (float*)std::malloc(n * sizeof(float));
    float* hs = (float*)std::malloc(n * sizeof(float));
    CK(cudaMemcpy(hd, dd, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs, ds, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
      printf("q=%-12.9g  1/q=%08x  sqrt=%08x\n",
             qs[i], fbits_host(hd[i]), fbits_host(hs[i]));
    CK(cudaFree(dq)); CK(cudaFree(dd)); CK(cudaFree(ds));
    std::free(hd); std::free(hs);
    return 0;
  }

  if (std::strcmp(mode, "atomden") == 0) {
    float* g; float* s;
    CK(cudaMalloc(&g, sizeof(float)));
    CK(cudaMalloc(&s, sizeof(float)));
    float zero = 0.0f;
    CK(cudaMemcpy(g, &zero, sizeof(float), cudaMemcpyHostToDevice));
    k_atomden<<<1, 32>>>(g, s);
    CK(cudaDeviceSynchronize());
    float hg = 0.0f, hs = 0.0f;
    CK(cudaMemcpy(&hg, g, sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hs, s, sizeof(float), cudaMemcpyDeviceToHost));
    printf("atomicAdd global 0 + 1e-40f : %08x  %.9g\n", fbits_host(hg), hg);
    printf("atomicAdd shared 0 + 1e-40f : %08x  %.9g\n", fbits_host(hs), hs);
    CK(cudaFree(g)); CK(cudaFree(s));
    return 0;
  }

  if (std::strcmp(mode, "sum") == 0) {
    const int runs = (argc > 2) ? std::atoi(argv[2]) : 5;
    const long n = 1L << 20;  // ~1M 个近似随机的 float，正负相消
    float* h = (float*)std::malloc(n * sizeof(float));
    for (long i = 0; i < n; ++i) {
      unsigned r = (unsigned)(i * 1664525u + 1013904223u);
      float u = (float)((r >> 8) & 0xFFFFFFu) / 16777216.0f;  // [0,1)
      h[i] = u * 2.0f - 1.0f;                                  // [-1,1)
    }

    double dsum = 0.0; float fsum = 0.0f;
    for (long i = 0; i < n; ++i) { dsum += (double)h[i]; fsum += h[i]; }
    printf("n=%ld\n", n);
    printf("host double sum   : %.17g\n", dsum);
    printf("host float  seq   : %08x  %.9g\n", fbits_host(fsum), fsum);

    float* d; CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    for (int k = 0; k < runs; ++k) {
      float zero = 0.0f;
      float* out; CK(cudaMalloc(&out, sizeof(float)));
      CK(cudaMemcpy(out, &zero, sizeof(float), cudaMemcpyHostToDevice));
      k_sum<<<4096 / 256, 256>>>(d, n, out);
      CK(cudaDeviceSynchronize());
      float r = 0.0f;
      CK(cudaMemcpy(&r, out, sizeof(float), cudaMemcpyDeviceToHost));
      printf("device atomic run %d : %08x  %.9g\n", k, fbits_host(r), r);
      CK(cudaFree(out));
    }
    CK(cudaFree(d)); std::free(h);
    return 0;
  }

  printf("unknown mode: %s\n", mode);
  return 2;
}
