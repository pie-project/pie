// The launchers for `attn/kimi_mla.cuh`, and nothing else.
//
// The two `__global__`s these fire used to live here. They are in the header
// now, which this file includes, so the ahead-of-time build compiles the same
// text NVRTC does -- one definition, not two that agree until one is edited.
// `new-horizon.md` §10.10 fixes the order of that split and says why.
//
// What is left is the host half: the extent arithmetic, the empty guards and
// the `<<<>>>`. Both are `LaunchRule`s already -- `Elementwise` and `Rms` --
// so the JIT reaches these kernels without any of it. The launchers stayed
// "until a later commit with a measurement behind it".
//
// THAT MEASUREMENT ARRIVED, and it applied to one of the two.
// `attn::kimi_split_kv_a_norm_bf16` is in `kernels_cuda_new::device`'s
// `JIT_DISPATCHED`, so `abi::emit_c_shim` skips its row and no
// `pie_k_attn_kimi_split_kv_a_norm_bf16` is generated -- and the shim entry
// was its only consumer. `scripts/csrc-reachability-audit.py` reports it
// unreachable for exactly that reason, and its launcher is deleted here.
// `kimi_split_q_b_bf16` is NOT routed to the JIT, still has a shim entry and
// is still reached, so it stays: the file is half migrated because the
// routing is, and that is the honest shape.
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

}  // namespace pie_cuda_driver::kernels::attn
