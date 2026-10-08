#include <cstdio>
__global__ void k(int* p){ p[threadIdx.x]=threadIdx.x; }
int main(){ int*d; cudaMalloc(&d,128*sizeof(int)); k<<<1,32>>>(d); cudaDeviceSynchronize(); int h[32]; cudaMemcpy(h,d,sizeof(h),cudaMemcpyDeviceToHost); printf("ok %d\n",h[31]); cudaFree(d); return 0; }
