// EXP-0009 — probe: does sm_89 (Ada) accept the sm_90 bulk-async-copy (TMA)
// instruction cp.async.bulk?  Expected: ptxas rejects it for sm_89, accepts
// it for sm_90. Compile both; capture errors. Machine: sm_89 only.
#include <cuda_runtime.h>

__global__ void tma_probe(const void *g_src, void *s_dst, unsigned size,
                          unsigned long long *s_mbar) {
  unsigned dst = (unsigned)__cvta_generic_to_shared(s_dst);
  unsigned mbar = (unsigned)__cvta_generic_to_shared(s_mbar);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n" ::"r"(dst),
      "l"(g_src), "r"(size), "r"(mbar)
      : "memory");
}

int main() { return 0; }
