// MLA paged-cache launchers.
//
// The device text -- `write_mla`, `mla_prepare<BLOCK_DIM>`, the two helpers
// they share and `kMaxRopePairs` -- moved to
// `crates/kernels-cuda-new/csrc/src/attn/mla_paged.cuh`, which this file
// includes. There is ONE text: the ahead-of-time build compiles it through
// this translation unit and NVRTC compiles the same header. §21.7 is what
// happens when there are two.
//
// Kept here: the `<<<>>>` themselves, `MlaCacheLayerView` unpacking, the
// `heads_per_block`/`q_blocks` grid arithmetic, the YaRN ramp bounds (a
// `__host__ __device__` helper called on the host), and `mla_prepare_supported`
// -- all host code, none of it compilable by NVRTC.
#include "pie_device.cuh"
#include "attn/mla_paged.cuh"
#include "attn/mla_paged.hpp"


#include "cuda_check.hpp"
#include "rope_device.cuh"

namespace pie_cuda_driver::kernels::attn {
void mla_prepare_bf16(
    MlaCacheLayerView layer,
    const void* kv_a,
    const void* kv_a_norm_weight,
    const void* q_b,
    void* kv_c,
    void* k_pe,
    void* q_nope,
    void* q_pe,
    const device::i32* positions,
    const device::u32* qo_indptr,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int heads,
    int qk_nope_head_dim,
    float eps,
    float theta,
    bool interleaved,
    int kv_a_row_stride,
    const YarnOriginalParams* yarn,
    cudaStream_t stream,
    const device::u8* row_valid)
{
    if (total_tokens <= 0) return;
    constexpr int BS = 256;
    const int kv_lora = layer.kv_lora_rank;
    const int rope = layer.qk_rope_head_dim;
    const int half = rope / 2;
    const int stride =
        kv_a_row_stride > 0 ? kv_a_row_stride : kv_lora + rope;
    // Match `kernels::rope::rope_bf16`'s head packing so the query lane has the same
    // shape of work per block that the standalone kernel had.
    const int heads_per_block = half >= BS ? 1 : (BS / half);
    const int q_blocks = (heads + heads_per_block - 1) / heads_per_block;
    float low_dim = 0.f, high_dim = 0.f;
    if (yarn != nullptr) {
        yarn_original_ramp_bounds(rope, theta, yarn->beta_fast,
                                  yarn->beta_slow,
                                  yarn->original_max_position,
                                  low_dim, high_dim);
    }
    dim3 grid(total_tokens, 1 + q_blocks);
    device::mla_prepare<BS><<<grid, BS, 0, stream>>>(
        static_cast<const device::bf16*>(kv_a),
        static_cast<const device::bf16*>(kv_a_norm_weight),
        static_cast<const device::bf16*>(q_b),
        static_cast<device::bf16*>(kv_c),
        static_cast<device::bf16*>(k_pe),
        static_cast<device::bf16*>(q_nope),
        static_cast<device::bf16*>(q_pe),
        static_cast<device::bf16*>(layer.ckv_pages),
        static_cast<device::bf16*>(layer.kpe_pages),
        positions, qo_indptr, kv_page_indices, kv_page_indptr,
        kv_last_page_lens, row_valid,
        num_requests, layer.page_size, heads, kv_lora, qk_nope_head_dim, rope,
        stride, eps, theta, interleaved, heads_per_block,
        yarn != nullptr ? yarn->factor : -1.f, low_dim, high_dim,
        yarn != nullptr ? yarn->attention_factor : 1.f);
    CUDA_CHECK(cudaGetLastError());
}

void write_mla_to_pages_bf16(
    void* ckv_pages,
    void* kpe_pages,
    const void* ckv_curr,
    const void* kpe_curr,
    const device::u32* qo_indptr,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim,
    cudaStream_t stream,
    const device::u8* row_valid)
{
    if (total_tokens <= 0) return;
    device::write_mla<<<total_tokens, 256, 0, stream>>>(
        static_cast<const device::bf16*>(ckv_curr),
        static_cast<const device::bf16*>(kpe_curr),
        static_cast<device::bf16*>(ckv_pages),
        static_cast<device::bf16*>(kpe_pages),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        row_valid,
        num_requests, page_size, kv_lora_rank, qk_rope_head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void write_mla_to_pages(
    MlaCacheLayerView layer,
    const void* ckv_curr,
    const void* kpe_curr,
    const device::u32* qo_indptr,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    cudaStream_t stream,
    const device::u8* row_valid)
{
    write_mla_to_pages_bf16(
        layer.ckv_pages, layer.kpe_pages, ckv_curr, kpe_curr,
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        total_tokens, num_requests, layer.page_size, layer.kv_lora_rank,
        layer.qk_rope_head_dim, stream, row_valid);
}

}  // namespace pie_cuda_driver::kernels::attn
