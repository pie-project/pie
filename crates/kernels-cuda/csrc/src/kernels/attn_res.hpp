#pragma once

// AttnRes — Kimi-K3's attention residuals.
//
// K3 does not add its sub-block outputs straight into one residual stream.
// It keeps a *stack* of block residuals: every `attn_res_block_size` layers
// the running sum is pushed onto the stack, and before every attention and
// every MLP the layer replaces its input with a softmax-weighted blend of
// the stack plus the running sum. The tail of the model blends once more
// through `output_attn_res_{norm,proj}`.
//
// The blend, over `v = [blocks... , prefix]` of `B + 1` rows per token
// (`_apply_attn_res` in Moonshot's `modeling_kimi_linear.py`):
//
//     k[j]     = v[j] * rsqrt(mean(v[j]^2) + eps)      (RMS, no weight yet)
//     score[j] = Σ_h k[j][h] * norm_w[h] * proj_w[h]
//     prob     = softmax(score)                        (over the B + 1 rows)
//     out      = Σ_j prob[j] * v[j]                    (un-normalised v)
//
// Note which tensor each half reads: the scores come from the *normalised*
// rows, the output from the raw ones. Everything is accumulated in fp32
// because `mean(v^2)` over 7168 bf16 channels is not representable otherwise.
//
// The block stack is stored **block-major**, `[B, T, H]`, so pushing a block
// is one contiguous `[T, H]` write instead of a strided scatter into a
// `[T, B, H]` tensor -- which is the layout the reference uses only because
// `torch.cat` on dim 1 is what expresses it in eager PyTorch.

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// `prefix`, `out` : [T, H] bf16 (may alias)
// `blocks`        : [B, block_rows, H] bf16 block-major; ignored when `B == 0`
// `block_rows`    : the *allocated* row count of a block, which is the
//                   workspace's max token count and not `T` -- a batch fills
//                   only the head of each block
// `norm_weight`   : [H] bf16 — the RMSNorm gain
// `proj_weight`   : [H] bf16 — the [1, H] score projection
//
// With `B == 0` the softmax is over a single row and `out == prefix`, which is
// what the tail blend of a model with no open blocks means.
void launch_attn_res_blend_bf16(
    const void* prefix,
    const void* blocks,
    const void* norm_weight,
    const void* proj_weight,
    void* out,
    int T, int B, int H,
    int block_rows,
    float eps,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
