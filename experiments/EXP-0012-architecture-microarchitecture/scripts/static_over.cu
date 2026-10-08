// EXP-0012 — static shared memory above the 48KB static limit.
// "__shared__ float big[26000]" = 104000 B. The hardware can hold ~100KB/SM,
// but STATIC shared memory per block is capped at 48KB (0xc000). Expect ptxas
// to reject this at compile time.
__global__ void too_much_static() {
  __shared__ float big[26000];
  big[threadIdx.x] = 1.0f;
  if (big[threadIdx.x] == 2.0f) big[0] = 0.0f;
}
