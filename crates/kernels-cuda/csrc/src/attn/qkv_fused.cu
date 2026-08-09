// Fused QKV epilogue launchers.
//
// The device text -- the three templates `qkv_decode_qk_norm_rope_write_kv`,
// `qkv_decode_qk_norm_rope_write_kv_warp` and
// `qkv_packed_qk_norm_rope_vnorm_write_kv` -- moved to
// `crates/kernels-cuda-new/csrc/src/attn/qkv_fused.cuh`, which this file
// includes. There is ONE text: the ahead-of-time build compiles it through
// this translation unit and NVRTC compiles the same header.
//
// Kept here: the five `<<<>>>`, the head-dim dispatch that chooses the warp
// form over the block form, and the `rope_table != nullptr` test that picks
// the `USE_ROPE_TABLE` arm. All host code.
#include "pie_device.cuh"
#include "attn/qkv_fused.cuh"
#include "attn/qkv_fused.hpp"


namespace pie_cuda_driver::kernels::attn {

// Shared dispatch for the fused decode epilogue: `num_requests` sizes
// the grid; `win` (nullable) is the Peel device window's prefix form —
// rows [0, win[0]) — read per-row so the launch shape stops depending
// on the split.
static void qkv_decode_fused_dispatch(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const device::i32* positions,
    const float* rope_table,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    const device::u32* w_page,
    const device::u32* w_off,
    const device::u8* row_valid,
    const device::u32* win,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream)
{
    if (num_requests == 0) return;
    constexpr int WARP_BLOCK = 256;
    const int total_units = num_requests * (num_q_heads + num_kv_heads);
    dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));
#define LAUNCH_QKV_DECODE_POST_WARP(HEAD_DIM_VALUE)                         \
    do {                                                                     \
        if (rope_table != nullptr) {                                         \
            device::qkv_decode_qk_norm_rope_write_kv_warp<                   \
                (HEAD_DIM_VALUE), true><<<warp_grid, WARP_BLOCK, 0, stream>>>( \
                    static_cast<const device::bf16*>(packed),               \
                    static_cast<device::bf16*>(q_out),                      \
                    static_cast<device::bf16*>(k_pages),                    \
                    static_cast<device::bf16*>(v_pages),                    \
                    static_cast<const device::bf16*>(q_weight),             \
                    static_cast<const device::bf16*>(k_weight),             \
                    positions, rope_table, kv_page_indices, kv_page_indptr,  \
                    kv_last_page_lens, w_page, w_off, row_valid, win,        \
                    num_requests, num_q_heads,                               \
                    num_kv_heads, page_size, hnd_layout, theta, eps);        \
        } else {                                                             \
            device::qkv_decode_qk_norm_rope_write_kv_warp<                   \
                (HEAD_DIM_VALUE), false><<<warp_grid, WARP_BLOCK, 0, stream>>>( \
                    static_cast<const device::bf16*>(packed),               \
                    static_cast<device::bf16*>(q_out),                      \
                    static_cast<device::bf16*>(k_pages),                    \
                    static_cast<device::bf16*>(v_pages),                    \
                    static_cast<const device::bf16*>(q_weight),             \
                    static_cast<const device::bf16*>(k_weight),             \
                    positions, rope_table, kv_page_indices, kv_page_indptr,  \
                    kv_last_page_lens, w_page, w_off, row_valid, win,        \
                    num_requests, num_q_heads,                               \
                    num_kv_heads, page_size, hnd_layout, theta, eps);        \
        }                                                                    \
    } while (0)
    if (head_dim == 64) {
        LAUNCH_QKV_DECODE_POST_WARP(64);
        return;
    }
    if (head_dim == 128) {
        LAUNCH_QKV_DECODE_POST_WARP(128);
        return;
    }
    if (head_dim == 256) {
        LAUNCH_QKV_DECODE_POST_WARP(256);
        return;
    }
#undef LAUNCH_QKV_DECODE_POST_WARP

    constexpr int BLOCK = 128;
    dim3 grid(num_requests, num_q_heads + num_kv_heads);
    if (rope_table != nullptr) {
        device::qkv_decode_qk_norm_rope_write_kv<BLOCK, true>
            <<<grid, BLOCK, 0, stream>>>(
                static_cast<const device::bf16*>(packed),
                static_cast<device::bf16*>(q_out),
                static_cast<device::bf16*>(k_pages),
                static_cast<device::bf16*>(v_pages),
                static_cast<const device::bf16*>(q_weight),
                static_cast<const device::bf16*>(k_weight),
                positions,
                rope_table,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                w_page,
                w_off,
                row_valid,
                win,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps);
    } else {
        device::qkv_decode_qk_norm_rope_write_kv<BLOCK, false>
            <<<grid, BLOCK, 0, stream>>>(
                static_cast<const device::bf16*>(packed),
                static_cast<device::bf16*>(q_out),
                static_cast<device::bf16*>(k_pages),
                static_cast<device::bf16*>(v_pages),
                static_cast<const device::bf16*>(q_weight),
                static_cast<const device::bf16*>(k_weight),
                positions,
                rope_table,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                w_page,
                w_off,
                row_valid,
                win,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps);
    }
}

void qkv_decode_qk_norm_rope_write_kv_bf16(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const device::i32* positions,
    const float* rope_table,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    const device::u32* w_page,
    const device::u32* w_off,
    const device::u8* row_valid,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream)
{
    qkv_decode_fused_dispatch(
        packed, q_out, k_pages, v_pages, q_weight, k_weight,
        positions, rope_table,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        w_page, w_off, row_valid, /*win=*/nullptr,
        num_requests, num_q_heads, num_kv_heads, head_dim,
        page_size, hnd_layout, theta, eps, stream);
}

void qkv_decode_qk_norm_rope_write_kv_bf16_devwin(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const device::i32* positions,
    const float* rope_table,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    const device::u32* w_page,
    const device::u32* w_off,
    const device::u8* row_valid,
    const device::u32* win_d,
    int n_max,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream)
{
    qkv_decode_fused_dispatch(
        packed, q_out, k_pages, v_pages, q_weight, k_weight,
        positions, rope_table,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        w_page, w_off, row_valid, win_d,
        n_max, num_q_heads, num_kv_heads, head_dim,
        page_size, hnd_layout, theta, eps, stream);
}

void qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const device::i32* positions,
    const device::u32* kv_page_indices,
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    const device::u8* row_valid,
    int num_rows,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream)
{
    if (num_rows == 0) return;
    constexpr int BLOCK = 256;
    dim3 grid(num_rows, num_q_heads + num_kv_heads);
    device::qkv_packed_qk_norm_rope_vnorm_write_kv<BLOCK>
        <<<grid, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(packed),
            static_cast<device::bf16*>(q_out),
            static_cast<device::bf16*>(k_pages),
            static_cast<device::bf16*>(v_pages),
            static_cast<const device::bf16*>(q_weight),
            static_cast<const device::bf16*>(k_weight),
            positions, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            row_valid,
            num_q_heads, num_kv_heads, head_dim, page_size, hnd_layout,
            theta, eps);
}

}  // namespace pie_cuda_driver::kernels::attn
