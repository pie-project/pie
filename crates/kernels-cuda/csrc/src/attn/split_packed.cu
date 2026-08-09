// The host launchers, and nothing else. Both `__global__`s live in
// `attn/split_packed.cuh` -- ONE definition, read by nvcc here and by NVRTC
// from the same text at run time.
//
// Neither has a row. `LaunchRule::SplitPacked` is the rule that fits and it
// is not ported by this backend's `runtime::launch` yet;
// `attn/split_packed.cuh` records what a port has to know, including why the
// rule's wider grid computes the same thing this narrower one does.
#include "attn/split_packed.cuh"
#include "attn/split_packed.hpp"


namespace pie_cuda_driver::kernels::attn {

namespace {

using bf16 = ::pie_cuda_driver::kernels::device::bf16;

constexpr int BLOCK = 256;

}  // namespace

void split_qkv_bf16_devwin(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    const device::u32* win_d,
    int n_max, int q_dim, int kv_dim,
    cudaStream_t stream)
{
    if (n_max <= 0) return;
    const int max_dim = q_dim > kv_dim ? q_dim : kv_dim;
    const int xblocks = (max_dim + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks, n_max);
    device::split_qkv_devwin<bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const bf16*>(packed),
        static_cast<bf16*>(q_out),
        static_cast<bf16*>(k_out),
        static_cast<bf16*>(v_out),
        win_d, q_dim, kv_dim);
}

void split_qkv_bf16(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    int n_tokens, int q_dim, int kv_dim,
    cudaStream_t stream)
{
    if (n_tokens == 0) return;
    const int max_dim = q_dim > kv_dim ? q_dim : kv_dim;
    const int xblocks = (max_dim + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks, n_tokens);
    device::split_qkv<bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const bf16*>(packed),
        static_cast<bf16*>(q_out),
        static_cast<bf16*>(k_out),
        static_cast<bf16*>(v_out),
        q_dim, kv_dim);
}

}  // namespace pie_cuda_driver::kernels::attn
