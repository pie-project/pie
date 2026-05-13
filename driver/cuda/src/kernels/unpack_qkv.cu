#include "kernels/unpack_qkv.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// Each thread copies one bf16 from the packed row to its q/k/v slice.
// Grid is 1-D over N*Htotal; arithmetic on `j` decides which slice the
// element lands in. Coalesced reads (consecutive threads → consecutive
// `j` → consecutive offsets in `qkv`), and coalesced writes within each
// slice because q/k/v are contiguous in the same dim order.
__global__ void unpack_qkv_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int N, int Hq, int Hk)
{
    const int Htotal = Hq + 2 * Hk;
    const long long total = static_cast<long long>(N) * Htotal;
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int n = static_cast<int>(idx / Htotal);
    const int j = static_cast<int>(idx % Htotal);
    const __nv_bfloat16 x = qkv[idx];
    if (j < Hq) {
        q[n * Hq + j] = x;
    } else if (j < Hq + Hk) {
        k[n * Hk + (j - Hq)] = x;
    } else {
        v[n * Hk + (j - Hq - Hk)] = x;
    }
}

}  // namespace

void launch_unpack_qkv_bf16(
    const void* qkv_packed, void* q, void* k, void* v,
    int N, int Hq, int Hk, cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * (Hq + 2 * Hk);
    const int threads = 256;
    const long long blocks_ll = (total + threads - 1) / threads;
    const int blocks = static_cast<int>(blocks_ll);
    unpack_qkv_bf16_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkv_packed),
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        static_cast<__nv_bfloat16*>(v),
        N, Hq, Hk);
}

}  // namespace pie_cuda_driver::kernels
