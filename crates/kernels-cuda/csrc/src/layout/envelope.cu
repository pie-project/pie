//===-- envelope.cu - the five envelope launchers --------------------===//
//
// Five host launchers and not one `__global__`: the device text is in
// `layout/envelope.cuh`, which this file includes so the archive and the JIT
// header set hold the SAME definition rather than two that drift.
//
// The launchers stay because `attn/kv_paged.cu` calls them and because six of
// the seven kernels state a `(page, kv_head)` grid no `LaunchRule` spells.
// Nothing about their callers changed.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names, out of the prelude.
#include "pie_device.cuh"
#include "layout/envelope.hpp"

// The `__global__`s these launchers fire. ONE definition of each.
#include "layout/envelope.cuh"

#include <cstdint>

namespace pie_cuda_driver::kernels::layout {

void launch_envelope_merge_written_bf16(
    const device::u16* k_curr,
    const device::u32* w_page,
    const device::u32* w_off,
    const device::u8* row_valid,
    device::u16* env_min,
    device::u16* env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    const dim3 grid(static_cast<unsigned>(num_tokens),
                    static_cast<unsigned>(num_kv_heads));
    const int threads = head_dim < 256 ? head_dim : 256;
    if (num_tokens <= device::kEnvelopeFuseMaxTokens) {
        device::merge_written_fused<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const device::bf16*>(k_curr),
            w_page, w_off, row_valid,
            reinterpret_cast<device::bf16*>(env_min),
        reinterpret_cast<device::bf16*>(env_max),
            num_tokens, num_kv_heads, head_dim);
        return;
    }
    device::reset_started_pages<<<grid, threads, 0, stream>>>(
        w_page, w_off, row_valid,
        reinterpret_cast<device::bf16*>(env_min),
        reinterpret_cast<device::bf16*>(env_max),
        num_tokens, num_kv_heads, head_dim);
    device::merge_written<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const device::bf16*>(k_curr),
        w_page, row_valid,
        reinterpret_cast<device::bf16*>(env_min),
        reinterpret_cast<device::bf16*>(env_max),
        num_tokens, num_kv_heads, head_dim);
}

void launch_envelope_seed_empty_bf16(
    device::u16* env_min,
    device::u16* env_max,
    int num_pages,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_pages <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    const device::usize n = static_cast<device::usize>(num_pages) *
                          static_cast<device::usize>(num_kv_heads) *
                          static_cast<device::usize>(head_dim);
    const int threads = 256;
    const device::usize blocks = (n + threads - 1) / threads;
    device::seed_empty<<<static_cast<unsigned>(blocks), threads, 0,
                                 stream>>>(
        reinterpret_cast<device::bf16*>(env_min),
        reinterpret_cast<device::bf16*>(env_max), n);
}

void launch_envelope_recompute_bf16(
    const device::u16* k_pages,
    const device::i32* page_live_lens,
    device::u16* env_min,
    device::u16* env_max,
    int num_pages,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_pages <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    const dim3 grid(static_cast<unsigned>(num_pages),
                    static_cast<unsigned>(num_kv_heads));
    const int threads = head_dim < 256 ? head_dim : 256;
    device::recompute<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const device::bf16*>(k_pages),
        page_live_lens,
        reinterpret_cast<device::bf16*>(env_min),
        reinterpret_cast<device::bf16*>(env_max),
        page_size, num_kv_heads, head_dim);
}

void launch_envelope_dot_f32(
    const float* q,
    const device::u16* env_min,
    const device::u16* env_max,
    float* score,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int p_max,
    int live_pages,
    cudaStream_t stream)
{
    if (p_max <= 0 || num_kv_heads <= 0) return;
    constexpr int BLOCK = 128;
    const dim3 grid(static_cast<unsigned>(p_max),
                    static_cast<unsigned>(num_kv_heads));
    device::dot<BLOCK><<<grid, BLOCK, 0, stream>>>(
        q,
        reinterpret_cast<const device::bf16*>(env_min),
        reinterpret_cast<const device::bf16*>(env_max),
        score,
        num_q_heads, num_kv_heads, head_dim, p_max, live_pages);
}

void launch_envelope_update_appended_bf16(
    const device::u16* k_pages,
    const device::u32* qo_indptr,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    device::u16* env_min,
    device::u16* env_max,
    int num_requests,
    int max_touched,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_requests <= 0 || max_touched <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(max_touched),
                    static_cast<unsigned>(num_kv_heads));
    const int threads = head_dim < 256 ? head_dim : 256;
    device::update_appended<device::bf16><<<grid, threads, 0, stream>>>(
        reinterpret_cast<const device::bf16*>(k_pages),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        reinterpret_cast<device::bf16*>(env_min),
        reinterpret_cast<device::bf16*>(env_max),
        num_requests, page_size, num_kv_heads, head_dim);
}

}  // namespace pie_cuda_driver::kernels::layout
