// The host launchers, and nothing else. All five `__global__`s and the two
// `__device__` helpers live in `attn/attention_naive.cuh` -- ONE definition,
// read by nvcc here and by NVRTC from the same text at run time.
//
// What stayed behind is everything a `LaunchRule` cannot say: the
// shared-memory budgets, which are sized on a KV extent; the `scale`, which
// is `1/sqrt(head_dim)` computed once on the host; and the FALLBACK in
// `attention_mtp_paged_history_bf16`, which chooses a different kernel when
// the global window will not fit in shared memory. A rule selects a
// rectangle, not a kernel.
//
// `<cuda_bf16.h>` and `<cmath>` went with the device text -- see the header
// for what NVRTC answered when it was asked for them. `sqrtf` here is the
// host's, out of `<cuda_runtime.h>` by way of the `.hpp`.
#include "attn/attention_naive.cuh"
#include "attn/attention_naive.hpp"

namespace pie_cuda_driver::kernels::attn {

namespace {

using bf16 = ::pie_cuda_driver::kernels::device::bf16;

constexpr int BLOCK = device::BLOCK;

}  // namespace

void attention_naive_bf16(
    const void* q, const void* k, const void* v,
    void* o,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(num_q_heads, num_tokens);
    dim3 block(BLOCK);
    const std::size_t shmem_bytes =
        sizeof(float) * (static_cast<std::size_t>(num_tokens) + BLOCK);

    device::attn_naive<bf16><<<grid, block, shmem_bytes, stream>>>(
        static_cast<const bf16*>(q),
        static_cast<const bf16*>(k),
        static_cast<const bf16*>(v),
        static_cast<bf16*>(o),
        num_tokens, num_q_heads, num_kv_heads, head_dim, scale);
}

void attention_mtp_history_bf16(
    const void* q,
    const void* k_history,
    const void* v_history,
    void* o,
    int num_tokens,
    int history_steps,
    int history_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || history_steps <= 0) return;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 grid(num_q_heads, num_tokens);
    dim3 block(BLOCK);
    const std::size_t shmem_bytes =
        sizeof(float) * (static_cast<std::size_t>(history_steps) + BLOCK);
    device::attn_mtp_history<bf16><<<grid, block, shmem_bytes, stream>>>(
        static_cast<const bf16*>(q),
        static_cast<const bf16*>(k_history),
        static_cast<const bf16*>(v_history),
        static_cast<bf16*>(o),
        num_tokens, history_steps, history_stride,
        num_q_heads, num_kv_heads, head_dim, scale);
}

void attention_mtp_paged_history_bf16(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* k_history,
    const void* v_history,
    void* o,
    const std::int32_t* position_ids,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int num_tokens,
    int history_steps,
    int history_stride,
    int max_global_tokens,
    int page_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool hnd_layout,
    bool global_cache_uses_prefix_position,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || history_steps <= 0) return;
    if (max_global_tokens <= 0) {
        attention_mtp_history_bf16(
            q, k_history, v_history, o, num_tokens, history_steps,
            history_stride, num_q_heads, num_kv_heads, head_dim, stream);
        return;
    }
    // Keep the reference kernel inside portable shared-memory limits. Long
    // contexts should use a FlashInfer-backed MTP decode path; until then,
    // fall back to local draft history instead of failing the launch.
    if (max_global_tokens + history_steps > 8192) {
        attention_mtp_history_bf16(
            q, k_history, v_history, o, num_tokens, history_steps,
            history_stride, num_q_heads, num_kv_heads, head_dim, stream);
        return;
    }
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 grid(num_q_heads, num_tokens);
    dim3 block(BLOCK);
    const std::size_t shmem_bytes = sizeof(float) *
        (static_cast<std::size_t>(max_global_tokens + history_steps) + BLOCK);
    device::attn_mtp_paged_history<bf16><<<grid, block, shmem_bytes, stream>>>(
        static_cast<const bf16*>(q),
        static_cast<const bf16*>(k_pages),
        static_cast<const bf16*>(v_pages),
        static_cast<const bf16*>(k_history),
        static_cast<const bf16*>(v_history),
        static_cast<bf16*>(o),
        position_ids, request_ids,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        num_tokens, history_steps, history_stride, max_global_tokens,
        page_size, num_q_heads, num_kv_heads, head_dim, hnd_layout, scale,
        global_cache_uses_prefix_position);
}

void mtp_shift_hidden_bf16(
    const void* target_hidden,
    const void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    void* out,
    int total_tokens,
    int num_requests,
    int hidden_size,
    cudaStream_t stream)
{
    if (total_tokens <= 0 || num_requests <= 0 || hidden_size <= 0 ||
        pending_hidden == nullptr) {
        return;
    }
    device::mtp_shift_hidden<bf16><<<total_tokens, BLOCK, 0, stream>>>(
        static_cast<const bf16*>(target_hidden),
        static_cast<const bf16*>(pending_hidden),
        qo_indptr, slot_ids,
        static_cast<bf16*>(out),
        num_requests, hidden_size);
}

void mtp_update_pending_hidden_bf16(
    const void* target_hidden,
    void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    int num_requests,
    int hidden_size,
    cudaStream_t stream)
{
    if (num_requests <= 0 || hidden_size <= 0 || pending_hidden == nullptr) {
        return;
    }
    device::mtp_update_pending_hidden<bf16><<<num_requests, BLOCK, 0, stream>>>(
        static_cast<const bf16*>(target_hidden),
        static_cast<bf16*>(pending_hidden),
        qo_indptr, slot_ids, num_requests, hidden_size);
}

}  // namespace pie_cuda_driver::kernels::attn
