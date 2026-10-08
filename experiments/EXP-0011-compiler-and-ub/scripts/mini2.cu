#include <cstdio>
#include <cuda_runtime.h>
static void mark(const char* s){ FILE* f=fopen("E:\\Project\\cuda-semantics\\tmp\\outputs\\D11\\mini2.trace","a"); if(f){fprintf(f,"%s\n",s); fclose(f);} }
__global__ void k(int* p){ p[threadIdx.x]=threadIdx.x; }
int main(){ mark("start"); int*d; cudaMalloc(&d,128*sizeof(int)); mark("malloc"); k<<<1,32>>>(d); cudaDeviceSynchronize(); mark("kernel-done"); int h[32]; cudaMemcpy(h,d,sizeof(h),cudaMemcpyDeviceToHost); mark("copy"); cudaFree(d); mark("end"); return 0; }
