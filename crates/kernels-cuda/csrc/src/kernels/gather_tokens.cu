#include "kernels/gather_tokens.hpp"

#include <cstdint>
#include <stdexcept>

namespace pie_cuda_driver::kernels {

namespace {

// int4-vectorized copy (8 bf16 / 16 B per element): one block per (op, layer),
// grid-stride over the op's contiguous span for K and V. Used when every span
// base + length is 8-bf16-aligned (`token_stride % 8 == 0`), which holds for
// head_dim ∈ {64,128,256,512}.
__global__ void gather_tokens_i4_kernel(
    int4* __restrict__ k,
    int4* __restrict__ v,
    const GatherTokenOp* __restrict__ ops,
    std::int64_t token_stride_i4,
    std::int64_t page_stride_i4,
    std::int64_t layer_stride_i4)
{
    const GatherTokenOp o = ops[blockIdx.x];
    const std::int64_t layer_off = static_cast<std::int64_t>(blockIdx.z) * layer_stride_i4;
    const std::int64_t span = static_cast<std::int64_t>(o.len) * token_stride_i4;
    const std::int64_t sbase = layer_off +
        static_cast<std::int64_t>(o.src_page) * page_stride_i4 +
        static_cast<std::int64_t>(o.src_off) * token_stride_i4;
    const std::int64_t dbase = layer_off +
        static_cast<std::int64_t>(o.dst_page) * page_stride_i4 +
        static_cast<std::int64_t>(o.dst_off) * token_stride_i4;
    for (std::int64_t i = threadIdx.x; i < span; i += blockDim.x) {
        k[dbase + i] = k[sbase + i];
        v[dbase + i] = v[sbase + i];
    }
}

// Scalar bf16 fallback for a non-8-aligned token stride.
__global__ void gather_tokens_u16_kernel(
    std::uint16_t* __restrict__ k,
    std::uint16_t* __restrict__ v,
    const GatherTokenOp* __restrict__ ops,
    std::int64_t token_stride,
    std::int64_t page_stride,
    std::int64_t layer_stride)
{
    const GatherTokenOp o = ops[blockIdx.x];
    const std::int64_t layer_off = static_cast<std::int64_t>(blockIdx.z) * layer_stride;
    const std::int64_t span = static_cast<std::int64_t>(o.len) * token_stride;
    const std::int64_t sbase = layer_off +
        static_cast<std::int64_t>(o.src_page) * page_stride +
        static_cast<std::int64_t>(o.src_off) * token_stride;
    const std::int64_t dbase = layer_off +
        static_cast<std::int64_t>(o.dst_page) * page_stride +
        static_cast<std::int64_t>(o.dst_off) * token_stride;
    for (std::int64_t i = threadIdx.x; i < span; i += blockDim.x) {
        k[dbase + i] = k[sbase + i];
        v[dbase + i] = v[sbase + i];
    }
}

void launch(
    std::uint16_t* k_pages, std::uint16_t* v_pages,
    const GatherTokenOp* ops, int num_ops,
    int num_layers, std::int64_t layer_stride_elems,
    int page_size, int num_kv_heads, int head_dim,
    cudaStream_t stream)
{
    if (num_ops <= 0 || num_layers <= 0) return;
    const std::int64_t token_stride =
        static_cast<std::int64_t>(num_kv_heads) * head_dim;
    const std::int64_t page_stride = token_stride * page_size;
    const int threads = 256;
    const dim3 grid(static_cast<unsigned>(num_ops), 1u,
                    static_cast<unsigned>(num_layers));

    if (token_stride % 8 == 0 && layer_stride_elems % 8 == 0) {
        gather_tokens_i4_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<int4*>(k_pages),
            reinterpret_cast<int4*>(v_pages),
            ops,
            token_stride / 8, page_stride / 8, layer_stride_elems / 8);
    } else {
        gather_tokens_u16_kernel<<<grid, threads, 0, stream>>>(
            k_pages, v_pages, ops,
            token_stride, page_stride, layer_stride_elems);
    }
}

}  // namespace

void launch_gather_tokens_bf16(
    std::uint16_t* k_pages, std::uint16_t* v_pages,
    const GatherTokenOp* ops, int num_ops,
    int page_size, int num_kv_heads, int head_dim,
    cudaStream_t stream)
{
    launch(k_pages, v_pages, ops, num_ops, /*num_layers=*/1,
           /*layer_stride_elems=*/0, page_size, num_kv_heads, head_dim, stream);
}

void launch_gather_tokens_bf16_layers(
    std::uint16_t* k_pages, std::uint16_t* v_pages,
    const GatherTokenOp* ops, int num_ops,
    int num_layers, std::int64_t layer_stride_elems,
    int page_size, int num_kv_heads, int head_dim,
    cudaStream_t stream)
{
    launch(k_pages, v_pages, ops, num_ops, num_layers, layer_stride_elems,
           page_size, num_kv_heads, head_dim, stream);
}

}  // namespace pie_cuda_driver::kernels
