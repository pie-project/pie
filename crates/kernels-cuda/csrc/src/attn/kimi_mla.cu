// The launchers for `attn/kimi_mla.cuh`, and nothing else.
//
// The two `__global__`s these fire used to live here. They are in the header
// now, which this file includes, so the ahead-of-time build compiles the same
// text NVRTC does -- one definition, not two that agree until one is edited.
// `new-horizon.md` §10.10 fixes the order of that split and says why.
//
// What is left is the host half: the extent arithmetic, the empty guards and
// the `<<<>>>`. Both are `LaunchRule`s already -- `Elementwise` and `Rms` --
// so the JIT reaches these kernels without any of it. The launchers stay
// because the ahead-of-time path is still the one every caller uses; deleting
// them is a later commit with a measurement behind it, not part of the split.
#include "pie_device.cuh"
#include "attn/kimi_mla.cuh"
#include "attn/kimi_mla.hpp"

namespace pie_cuda_driver::kernels::attn {

namespace {

constexpr int BLOCK = 256;

}  // namespace

void kimi_split_q_b_bf16(
    const void* q_b,
    void* q_nope,
    void* q_pe,
    int tokens,
    int heads,
    int qk_nope_dim,
    int qk_rope_dim,
    cudaStream_t stream)
{
    const int total = tokens * heads * (qk_nope_dim + qk_rope_dim);
    if (total <= 0) return;
    device::split_q_b<device::bf16><<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(q_b),
        static_cast<device::bf16*>(q_nope),
        static_cast<device::bf16*>(q_pe),
        total, heads, qk_nope_dim, qk_rope_dim);
}

void kimi_split_kv_a_norm_bf16(
    const void* kv_a,
    const void* norm_weight,
    void* kv_c,
    void* k_pe,
    int tokens,
    int kv_lora_rank,
    int qk_rope_dim,
    float eps,
    cudaStream_t stream,
    int src_row_stride)
{
    if (tokens <= 0) return;
    constexpr int BS = 256;
    const int stride =
        src_row_stride > 0 ? src_row_stride : kv_lora_rank + qk_rope_dim;
    device::split_kv_a_norm<device::bf16, BS><<<tokens, BS, 0, stream>>>(
        static_cast<const device::bf16*>(kv_a),
        static_cast<const device::bf16*>(norm_weight),
        static_cast<device::bf16*>(kv_c),
        static_cast<device::bf16*>(k_pe),
        kv_lora_rank, qk_rope_dim, stride, eps);
}

}  // namespace pie_cuda_driver::kernels::attn
