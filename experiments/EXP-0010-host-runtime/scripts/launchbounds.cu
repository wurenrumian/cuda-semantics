// EXP-0010 — __launch_bounds__ becomes PTX entry directives.
// This bridges the HOST launch contract (block dims, occupancy hints) to the
// DEVICE module: .maxntid / .reqntid / .minnctapersm in the .entry header.
// Dump: nvcc -arch=sm_89 -ptx launchbounds.cu -o launchbounds.ptx
__global__ void __launch_bounds__(256, 2) bounded(int *out) {
  out[threadIdx.x] = threadIdx.x;
}
__global__ void unbounded(int *out) {
  out[threadIdx.x] = threadIdx.x;
}
