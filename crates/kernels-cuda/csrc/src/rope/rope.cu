//===-- rope.cu - the rotary family's launchers ------------------------===//
//
// What is left of this file after the migration: twelve host launchers and
// not one `__global__`. Every kernel they fire is in `rope/rope.cuh`, which
// this file includes -- a SPLIT and not a copy, because two definitions that
// agree today drift tomorrow and each stays right for whichever half of the
// tests exercises it. `norm/altup_aux` shipped exactly that for a release.
//
// The file survives because the JIT does not host these launchers yet. Ten of
// the twelve state a geometry no `LaunchRule` spells -- a `dim3(token, head)`
// grid, a dynamic shared allocation sized off `head_dim`, or a YaRN ramp
// computed on the host before the launch -- so they keep their `<<<>>>` and
// their callers keep calling them. `kernels-cuda-new/src/families/rope.rs`
// names the two that DID become rows and says why the rest did not.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names. What used to be
// `<cuda_bf16.h>` and `<cstdint>`.
#include "pie_device.cuh"
#include "rope/rope.hpp"

// The `__global__`s these launchers fire. ONE definition of each, here and in
// the JIT's header set both.
#include "rope/rope.cuh"

namespace pie_cuda_driver::kernels::rope {

void qk_rmsnorm_mrope_bf16(
    void* q, void* k,
    const void* q_weight, const void* k_weight,
    const device::i32* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_q_heads + num_kv_heads <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    device::qk_rmsnorm_rotate_mrope<BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        static_cast<const device::bf16*>(q_weight),
        static_cast<const device::bf16*>(k_weight),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, eps,
        mrope_section_t, mrope_section_h, mrope_section_w);
}

void rope_standard_table(
    const device::i32* positions,
    float* table,
    int num_tokens,
    int head_dim,
    float theta,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || table == nullptr) return;
    constexpr int BLOCK = 128;
    device::standard_table<device::i32><<<num_tokens, BLOCK, 0, stream>>>(
        positions, table, head_dim, theta);
}

void rope_bf16(
    void* q, void* k,
    const device::i32* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    cudaStream_t stream,
    bool interleaved)
{
    constexpr int BLOCK = 256;
    // 32 KB caps the table at head_dim 8192; past that the pairs are recomputed.
    constexpr int kMaxCachedPairs = 4096;
    const int half = head_dim / 2;
    if (half <= 0) return;
    const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
    const device::usize smem = static_cast<device::usize>(cache_pairs) * 2 * sizeof(float);
    // Splitting the heads across blockIdx.y keeps every SM fed at decode, where
    // `num_tokens` is 1 and a 1-D grid would run a single block on 148 SMs.
    const int total_heads = num_q_heads + num_kv_heads;
    const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
    dim3 grid(num_tokens, (total_heads + heads_per_block - 1) / heads_per_block);
    dim3 block(BLOCK);
    device::rotate<false, false><<<grid, block, smem, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, interleaved, cache_pairs,
        heads_per_block,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        0, 0);
}

void rope_write_kv_bf16(
    void* q, void* k, const void* v,
    const device::i32* positions,
    void* k_pages, void* v_pages,
    const device::u32* qo_indptr,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    const device::u8* row_valid,
    int num_tokens, int num_requests, int page_size,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, bool hnd_layout,
    cudaStream_t stream, bool interleaved)
{
    constexpr int BLOCK = 256;
    constexpr int kMaxCachedPairs = 4096;
    const int half = head_dim / 2;
    if (half <= 0 || num_tokens <= 0) return;
    const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
    const device::usize smem =
        static_cast<device::usize>(cache_pairs) * 2 * sizeof(float);
    const int total_heads = num_q_heads + num_kv_heads;
    const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
    dim3 grid(num_tokens, (total_heads + heads_per_block - 1) / heads_per_block);
    dim3 block(BLOCK);
    auto launch = [&](auto hnd) {
        device::rotate<true, decltype(hnd)::value>
            <<<grid, block, smem, stream>>>(
                static_cast<device::bf16*>(q),
                static_cast<device::bf16*>(k),
                positions,
                num_q_heads, num_kv_heads, head_dim, theta, interleaved,
                cache_pairs, heads_per_block,
                static_cast<const device::bf16*>(v),
                static_cast<device::bf16*>(k_pages),
                static_cast<device::bf16*>(v_pages),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                row_valid, num_requests, page_size);
    };
    if (hnd_layout) launch(device::true_type{});
    else launch(device::false_type{});
}

void qk_rmsnorm_rope_bf16_devwin(
    void* q, void* k,
    const void* q_weight, const void* k_weight,
    const device::i32* positions,
    const device::u32* win_d,
    int n_max,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    cudaStream_t stream)
{
    if (n_max <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(n_max, num_q_heads + num_kv_heads);
    device::qk_rmsnorm_rotate_devwin<BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        static_cast<const device::bf16*>(q_weight),
        static_cast<const device::bf16*>(k_weight),
        positions, win_d,
        num_q_heads, num_kv_heads, head_dim, theta, eps);
}

void qk_rmsnorm_rope_bf16(
    void* q, void* k,
    const void* q_weight, const void* k_weight,
    const device::i32* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    cudaStream_t stream)
{
    // Zero-row guard: a grid.x of 0 is an invalid launch that poisons
    // the sticky error state (caught by the peel-window A/B harness's
    // empty-window case; production call sites gate on row counts).
    if (num_tokens <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    device::qk_rmsnorm_rotate<BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        static_cast<const device::bf16*>(q_weight),
        static_cast<const device::bf16*>(k_weight),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, eps);
}

void qk_rmsnorm_rope_bf16_rounded(
    void* q, void* k,
    const void* q_weight, const void* k_weight,
    const device::i32* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_q_heads + num_kv_heads <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    device::qk_rmsnorm_rotate_rounded<BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        static_cast<const device::bf16*>(q_weight),
        static_cast<const device::bf16*>(k_weight),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, eps);
}

// ── YaRN variant ────────────────────────────────────────────────────────────

void rope_yarn_bf16(
    void* q, void* k,
    const device::i32* positions,
    int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_freq_factor, float high_freq_factor,
    int original_max_position,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int half = head_dim / 2;
    if (half <= 0) return;
    const int total_heads = num_q_heads + num_kv_heads;
    const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
    const dim3 grid(num_tokens,
                    (total_heads + heads_per_block - 1) / heads_per_block);
    device::rotate_yarn<<<grid, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim,
        theta, factor, low_freq_factor, high_freq_factor,
        static_cast<float>(original_max_position),
        heads_per_block);
}

// ── Original YaRN variant (OLMo-3, gpt-oss) ───────────────────────────────

void rope_yarn_original_bf16(
    void* q, void* k,
    const device::i32* positions,
    int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float beta_fast, float beta_slow,
    float attention_factor,
    int original_max_position,
    cudaStream_t stream,
    bool interleaved)
{
    // correction_dim(rot) = head_dim * ln(max_pos / (rot * 2π)) / (2 * ln(theta)).
    // beta_slow → "low rotation count" → larger correction_dim → upper bound on
    // the ramp (above this, fully interpolated). beta_fast → smaller
    // correction_dim → lower bound (below this, fully extrapolated). HF clamps
    // to [0, head_dim/2 - 1].
    float low_dim = 0.f, high_dim = 0.f;
    yarn_original_ramp_bounds(head_dim, theta, beta_fast, beta_slow,
                              original_max_position, low_dim, high_dim);

    constexpr int BLOCK = 256;
    // One block per token leaves 147 of the B200's 148 SMs idle during decode,
    // where `num_tokens` is 1. Give each block a slice of the heads instead, so
    // the grid grows with the head count rather than the batch, and each thread
    // owns exactly one element -- one load/store round trip rather than a chain
    // of them.
    constexpr int kMaxCachedPairs = 4096;   // 32 KB of float2
    const int half = head_dim / 2;
    if (half <= 0) return;
    const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
    const int total_heads = num_q_heads + num_kv_heads;
    const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
    const dim3 grid(num_tokens,
                    (total_heads + heads_per_block - 1) / heads_per_block);
    const device::usize shared =
        static_cast<device::usize>(cache_pairs) * sizeof(float2);
    device::rotate_yarn_original<<<grid, BLOCK, shared, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim,
        theta, factor, low_dim, high_dim, attention_factor, interleaved,
        heads_per_block, cache_pairs);
}

// ── Partial rotary (Gemma-4 full-attention layers) ─────────────────────────

void rope_partial_bf16(
    void* q, void* k,
    const device::i32* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    device::rotate_partial<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        positions,
        0,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, theta);
}

void rope_partial_bf16_position_delta(
    void* q, void* k,
    const device::i32* positions,
    int position_delta,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    device::rotate_partial<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        positions,
        position_delta,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, theta);
}

void rope_partial_last_bf16(
    void* q, void* k,
    const device::i32* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream,
    bool inverse,
    bool interleaved,
    float yarn_factor,
    float yarn_beta_fast,
    float yarn_beta_slow,
    int   yarn_original_max_position)
{
    // Same ramp as `rope_yarn_original_bf16`, but the correction range
    // is over `rotary_dim` (the rotated slice), not the full head_dim.
    float low_dim = 0.f, high_dim = 0.f;
    if (yarn_factor > 1.f && yarn_original_max_position > 0) {
        constexpr float TWO_PI = 6.2831853071795864769f;
        const float ln_theta = logf(theta);
        auto corr_dim = [&](float rot) -> float {
            return rotary_dim * logf(static_cast<float>(yarn_original_max_position) /
                                     (rot * TWO_PI)) / (2.f * ln_theta);
        };
        low_dim  = floorf(corr_dim(yarn_beta_fast));
        high_dim = ceilf(corr_dim(yarn_beta_slow));
        if (low_dim < 0.f) low_dim = 0.f;
        const float max_pair = static_cast<float>(rotary_dim / 2) - 1.f;
        if (high_dim > max_pair) high_dim = max_pair;
        if (high_dim < low_dim)  high_dim = low_dim;
    }
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    device::rotate_partial_last<<<grid, block, 0, stream>>>(
        static_cast<device::bf16*>(q),
        static_cast<device::bf16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, theta, inverse,
        interleaved, yarn_factor, low_dim, high_dim);
}

}  // namespace pie_cuda_driver::kernels::rope
