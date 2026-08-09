// The host half of the paged KV cache, and nothing else.
//
// The fourteen `__global__`s this file used to hold -- six write forms, four
// dequantisers, a cell move, a device-window write and the two page-view
// builders -- and the four `__device__` helpers they call live in
// `attn/kv_paged.cuh`, which is what the include below reads. ONE text, read
// by nvcc here and by NVRTC from the same bytes at run time.
//
// It was two. This file kept its own copy of all fourteen while the header
// carried fourteen more with the same bodies, and every gate was green
// because a split RENAMES -- `write_kv_kernel` became `write_kv`, in another
// namespace -- so the name-comparing gate could not see them
// (`new-horizon.md` §21.7). `a_split_file_uses_the_header_it_was_split_into`
// asks the question that needs no names and is what closed it.
//
// # The launches are unchanged, and that is the claim
//
// Twenty-two `<<<>>>`, the same twenty-two, each with the grid, block,
// shared-memory size and stream it had. Nine of the header's fourteen are
// templates; the argument every launch below passes is the one that
// reproduces the archive's kernel, measured rather than inferred:
//
// * `HND_LAYOUT` and `UseFp8` were already the archive's own template
//   parameters and each call site already chose an arm -- those carried over
//   spelling for spelling.
// * The three `template <class T>` dequantisers were PLAIN here, writing
//   `__nv_bfloat16`, so `T` is `device::bf16` and nothing else. It is stated
//   explicitly at each launch rather than defaulted: a wrong arm compiles,
//   runs, and is numerically plausible (§18.4 measured one at 99.83% of the
//   right answer), so the argument is written where a reader can check it
//   against the cast beside it.
//
// All fourteen were compared on an L40S against the archive's bodies taken
// verbatim from git, same input, `memcmp` on the destination: zero bytes
// differ on twenty instantiations, and every `bool` and `class T` parameter
// has a negative control on the other arm that DOES differ.
//
// `fp8_kind` stays a runtime `__nv_fp8_interpretation_t` argument on the two
// kernels that take one, for the reason the header gives: as a template
// parameter with a default, an `__NV_E5M2` page would decode as `__NV_E4M3`
// and be wrong plausibly. Both interpretations are in the parity set.
//
// # This file is the header's only includer in the archive, and must stay so
//
// Five of the fourteen are not templates -- `write_kv_fp8_per_tensor`,
// `write_kv_fp4_block`, `dequant_fp8_pages_active`, `build_window_page_view`
// and `build_full_split_view`. A `.cuh` holding a non-template `__global__`
// can be included by exactly one translation unit: the host stub and the
// function both take external linkage, so a second includer is a hard
// `multiple definition` at link EVEN IF IT NEVER LAUNCHES IT (§21.6). A
// second consumer means templating those five first, which is a body change
// and needs its own parity evidence.
#include "attn/kv_paged.cuh"
#include "attn/kv_paged.hpp"

#include <cuda_fp8.h>
#include <stdexcept>

#include "cuda_check.hpp"
#include "layout/envelope.hpp"

namespace pie_cuda_driver::kernels::attn {

void write_kv_to_pages_bf16(
    void* k_pages, void* v_pages,
    const void* k_curr, const void* v_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim,
    bool hnd_layout,
    cudaStream_t stream,
    const std::uint8_t* row_valid,
    int first_token)
{
    constexpr int BLOCK = 256;
    const int launch_tokens = total_tokens - first_token;
    if (launch_tokens <= 0) return;
    if (hnd_layout) {
        device::write_kv<true><<<launch_tokens, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(k_pages),
            static_cast<device::bf16*>(v_pages),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            row_valid, /*win=*/nullptr,
            num_requests, page_size, num_kv_heads, head_dim,
            first_token);
    } else {
        device::write_kv<false><<<launch_tokens, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(k_pages),
            static_cast<device::bf16*>(v_pages),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            row_valid, /*win=*/nullptr,
            num_requests, page_size, num_kv_heads, head_dim,
            first_token);
    }
}

void write_kv_to_pages(
    KvCacheLayerView layer,
    const void* k_curr,
    const void* v_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    cudaStream_t stream,
    const std::uint8_t* row_valid,
    int first_token)
{
    const int page_size = layer.page_size;
    const int num_kv_heads = layer.num_kv_heads;
    const int head_dim = layer.head_dim;
    // A non-zero `first_token` means the leading tokens' K/V were written by
    // a fused kernel that only exists for the native-bf16 cache; on any other
    // scheme a partial write here would leave the prefix rows holding garbage
    // from `k_curr` rows nobody filled. Refuse loudly.
    if (first_token != 0 && !layer.is_native_bf16()) {
        throw std::runtime_error(
            "write_kv_to_pages: partial (first_token) writes require the "
            "native bf16 cache");
    }
    if (layer.is_native_bf16()) {
        write_kv_to_pages_bf16(
            layer.k_pages, layer.v_pages, k_curr, v_curr,
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            total_tokens, num_requests, page_size, num_kv_heads, head_dim,
            layer.hnd_layout, stream, row_valid, first_token);
        // Quest maintenance rides the append: the pages this fire just grew are
        // exactly the ones whose envelopes went stale, and the same stream
        // orders the refresh after the write. Opt-in -- `has_envelopes()` is
        // false unless a program declared it needs them.
        if (layer.has_envelopes() && !layer.hnd_layout && total_tokens > 0) {
            kernels::layout::launch_envelope_update_appended_bf16(
                static_cast<const std::uint16_t*>(layer.k_pages),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                layer.k_env_min, layer.k_env_max, num_requests,
                (total_tokens + page_size - 1) / page_size + num_requests,
                page_size, num_kv_heads, head_dim, stream);
        }
        return;
    }

    constexpr int BLOCK = 256;
    switch (layer.scheme) {
        case KvCacheScheme::Fp8PerTensor: {
            const auto fp8_kind = layer.storage_dtype == DType::FP8_E5M2
                ? __NV_E5M2
                : __NV_E4M3;
            device::write_kv_fp8_per_tensor<<<total_tokens, BLOCK, 0, stream>>>(
                static_cast<const device::bf16*>(k_curr),
                static_cast<const device::bf16*>(v_curr),
                static_cast<__nv_fp8_storage_t*>(layer.k_pages),
                static_cast<__nv_fp8_storage_t*>(layer.v_pages),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim, fp8_kind);
            break;
        }
        case KvCacheScheme::Int8PerTokenHead: {
            const dim3 grid(total_tokens, num_kv_heads);
            const std::size_t shmem = 2 * (BLOCK / 32) * sizeof(float);
            device::write_kv_per_token_head<false><<<grid, BLOCK, shmem, stream>>>(
                static_cast<const device::bf16*>(k_curr),
                static_cast<const device::bf16*>(v_curr),
                layer.k_pages, layer.v_pages,
                static_cast<float*>(layer.k_scales),
                static_cast<float*>(layer.v_scales),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim);
            break;
        }
        case KvCacheScheme::Fp8PerTokenHead: {
            const dim3 grid(total_tokens, num_kv_heads);
            const std::size_t shmem = 2 * (BLOCK / 32) * sizeof(float);
            device::write_kv_per_token_head<true><<<grid, BLOCK, shmem, stream>>>(
                static_cast<const device::bf16*>(k_curr),
                static_cast<const device::bf16*>(v_curr),
                layer.k_pages, layer.v_pages,
                static_cast<float*>(layer.k_scales),
                static_cast<float*>(layer.v_scales),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim);
            break;
        }
        case KvCacheScheme::Fp4Block: {
            const int block_size = layer.block_size > 0
                ? layer.block_size
                : 16;
            const int blocks = (head_dim + block_size - 1) / block_size;
            const dim3 grid(total_tokens, num_kv_heads, blocks);
            device::write_kv_fp4_block<<<grid, 32, 0, stream>>>(
                static_cast<const device::bf16*>(k_curr),
                static_cast<const device::bf16*>(v_curr),
                static_cast<std::uint8_t*>(layer.k_pages),
                static_cast<std::uint8_t*>(layer.v_pages),
                static_cast<float*>(layer.k_scales),
                static_cast<float*>(layer.v_scales),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim, block_size);
            break;
        }
        case KvCacheScheme::Native:
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void write_kv_to_pages_at_positions_bf16(
    KvCacheLayerView layer,
    const void* k_curr,
    const void* v_curr,
    const std::int32_t* positions,
    int position_delta,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    int total_tokens,
    int num_requests,
    cudaStream_t stream)
{
    if (!layer.is_native_bf16()) {
        throw std::runtime_error(
            "write_kv_to_pages_at_positions_bf16 requires native bf16 KV cache");
    }
    constexpr int BLOCK = 256;
    if (layer.hnd_layout) {
        device::write_kv_at_positions<true><<<total_tokens, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            positions, position_delta, qo_indptr, kv_page_indices,
            kv_page_indptr, num_requests, layer.page_size,
            layer.num_kv_heads, layer.head_dim);
    } else {
        device::write_kv_at_positions<false><<<total_tokens, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            positions, position_delta, qo_indptr, kv_page_indices,
            kv_page_indptr, num_requests, layer.page_size,
            layer.num_kv_heads, layer.head_dim);
    }
    CUDA_CHECK(cudaGetLastError());
}

void write_kv_explicit_bf16_devwin(
    KvCacheLayerView layer,
    const void* k_curr,
    const void* v_curr,
    const std::uint32_t* w_page,
    const std::uint32_t* w_off,
    const std::uint32_t* win_d,
    int n_max,
    cudaStream_t stream,
    const std::uint8_t* row_valid)
{
    if (!layer.is_native_bf16()) {
        throw std::runtime_error(
            "write_kv_explicit_bf16_devwin requires native bf16 KV cache");
    }
    if (n_max <= 0) return;
    // Envelope maintenance (quest) is NOT wired on this variant yet —
    // the campaign converts it when a windowed producer needs it; until
    // then a caller with envelopes must stay on the host-window form.
    if (layer.has_envelopes()) {
        throw std::runtime_error(
            "write_kv_explicit_bf16_devwin: envelope maintenance not yet "
            "windowed — use the host-window form");
    }
    constexpr int BLOCK = 256;
    if (layer.hnd_layout) {
        device::write_kv_explicit_devwin<true><<<n_max, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            w_page, w_off, row_valid, win_d, n_max, layer.page_size,
            layer.num_kv_heads, layer.head_dim);
    } else {
        device::write_kv_explicit_devwin<false><<<n_max, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            w_page, w_off, row_valid, win_d, n_max, layer.page_size,
            layer.num_kv_heads, layer.head_dim);
    }
    CUDA_CHECK(cudaGetLastError());
}

void write_kv_to_pages_bf16_devwin(
    KvCacheLayerView layer,
    const void* k_curr,
    const void* v_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* win_d,
    int n_max,
    int num_requests,
    cudaStream_t stream,
    const std::uint8_t* row_valid)
{
    if (!layer.is_native_bf16()) {
        throw std::runtime_error(
            "write_kv_to_pages_bf16_devwin requires native bf16 KV cache "
            "(the same argument as the host first_token form)");
    }
    if (n_max <= 0) return;
    // Envelope maintenance (quest) is NOT wired on this variant yet —
    // same disposition as the explicit devwin write.
    if (layer.has_envelopes()) {
        throw std::runtime_error(
            "write_kv_to_pages_bf16_devwin: envelope maintenance not yet "
            "windowed — use the host-window form");
    }
    constexpr int BLOCK = 256;
    if (layer.hnd_layout) {
        device::write_kv<true><<<n_max, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            row_valid, win_d,
            num_requests, layer.page_size, layer.num_kv_heads,
            layer.head_dim, /*first_token=*/0);
    } else {
        device::write_kv<false><<<n_max, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            row_valid, win_d,
            num_requests, layer.page_size, layer.num_kv_heads,
            layer.head_dim, /*first_token=*/0);
    }
    CUDA_CHECK(cudaGetLastError());
}

void write_kv_explicit_bf16(
    KvCacheLayerView layer,
    const void* k_curr,                 // [LANES, h_kv, d]
    const void* v_curr,
    const std::uint32_t* w_page,        // [LANES] PHYSICAL page id per lane
    const std::uint32_t* w_off,         // [LANES] offset-in-page per lane
    int B,
    cudaStream_t stream,
    const std::uint8_t* row_valid)
{
    if (!layer.is_native_bf16()) {
        throw std::runtime_error(
            "write_kv_explicit_bf16 requires native bf16 KV cache");
    }
    if (B <= 0) return;
    constexpr int BLOCK = 256;
    if (layer.hnd_layout) {
        device::write_kv_explicit<true><<<B, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            w_page, w_off, row_valid, B, layer.page_size, layer.num_kv_heads,
            layer.head_dim);
    } else {
        device::write_kv_explicit<false><<<B, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(k_curr),
            static_cast<const device::bf16*>(v_curr),
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            w_page, w_off, row_valid, B, layer.page_size, layer.num_kv_heads,
            layer.head_dim);
    }
    CUDA_CHECK(cudaGetLastError());
    // Quest maintenance rides this append too. The CSR-derived path in
    // `write_kv_to_pages` cannot be reused: there is no page list here,
    // only the per-token descriptor the program wrote. Opt-in on
    // `has_envelopes()`, same stream, so the refresh is ordered after the
    // write it describes.
    if (layer.has_envelopes() && !layer.hnd_layout) {
        kernels::layout::launch_envelope_merge_written_bf16(
            static_cast<const std::uint16_t*>(k_curr),
            w_page, w_off, row_valid, layer.k_env_min, layer.k_env_max,
            B, layer.num_kv_heads, layer.head_dim, stream);
        CUDA_CHECK(cudaGetLastError());
    }
}

void copy_kv_cells_bf16(
    KvCacheLayerView layer,
    const std::uint32_t* dst_page,      // [N] PHYSICAL page id per cell
    const std::uint32_t* dst_off,       // [N] offset-in-page per cell
    const std::uint32_t* src_page,      // [N] PHYSICAL page id per cell
    const std::uint32_t* src_off,       // [N] offset-in-page per cell
    int N,
    cudaStream_t stream)
{
    if (!layer.is_native_bf16()) {
        throw std::runtime_error(
            "copy_kv_cells_bf16 requires native bf16 KV cache");
    }
    if (N <= 0) return;
    constexpr int BLOCK = 256;
    if (layer.hnd_layout) {
        device::copy_kv_cells<true><<<N, BLOCK, 0, stream>>>(
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            dst_page, dst_off, src_page, src_off, N, layer.page_size,
            layer.num_kv_heads, layer.head_dim);
    } else {
        device::copy_kv_cells<false><<<N, BLOCK, 0, stream>>>(
            static_cast<device::bf16*>(layer.k_pages),
            static_cast<device::bf16*>(layer.v_pages),
            dst_page, dst_off, src_page, src_off, N, layer.page_size,
            layer.num_kv_heads, layer.head_dim);
    }
    CUDA_CHECK(cudaGetLastError());
}

void dequant_kv_cache_layer_to_bf16_active(
    KvCacheLayerView layer,
    const std::uint32_t* kv_page_indices,
    int num_pages_in_batch,
    cudaStream_t stream)
{
    if (layer.is_native_bf16() || num_pages_in_batch <= 0) return;
    constexpr int BLOCK = 256;
    const int page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
    const long long logical_n =
        static_cast<long long>(num_pages_in_batch) * page_elems;
    const auto blocks = static_cast<unsigned>((logical_n + BLOCK - 1) / BLOCK);

    switch (layer.scheme) {
        case KvCacheScheme::Fp8PerTensor: {
            const auto fp8_kind = layer.storage_dtype == DType::FP8_E5M2
                ? __NV_E5M2
                : __NV_E4M3;
            device::dequant_fp8_pages_active<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const __nv_fp8_storage_t*>(layer.k_pages),
                static_cast<const __nv_fp8_storage_t*>(layer.v_pages),
                static_cast<device::bf16*>(layer.k_bf16_pages),
                static_cast<device::bf16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, page_elems, fp8_kind);
            break;
        }
        case KvCacheScheme::Fp8PerTokenHead:
            device::dequant_fp8_per_token_head_pages_active<device::bf16>
                <<<blocks, BLOCK, 0, stream>>>(
                static_cast<const __nv_fp8_storage_t*>(layer.k_pages),
                static_cast<const __nv_fp8_storage_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<device::bf16*>(layer.k_bf16_pages),
                static_cast<device::bf16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim);
            break;
        case KvCacheScheme::Int8PerTokenHead:
            device::dequant_int8_per_token_head_pages_active<device::bf16>
                <<<blocks, BLOCK, 0, stream>>>(
                static_cast<const std::int8_t*>(layer.k_pages),
                static_cast<const std::int8_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<device::bf16*>(layer.k_bf16_pages),
                static_cast<device::bf16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim);
            break;
        case KvCacheScheme::Fp4Block: {
            const int block_size = layer.block_size > 0
                ? layer.block_size
                : 16;
            device::dequant_fp4_pages_active<device::bf16>
                <<<blocks, BLOCK, 0, stream>>>(
                static_cast<const std::uint8_t*>(layer.k_pages),
                static_cast<const std::uint8_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<device::bf16*>(layer.k_bf16_pages),
                static_cast<device::bf16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim, block_size);
            break;
        }
        case KvCacheScheme::Native:
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void build_window_page_view(
    const std::uint32_t* src_indices,
    const std::uint32_t* src_indptr,
    int keep_pages,
    std::uint32_t* dst_indptr,
    std::uint32_t* dst_indices,
    int R,
    cudaStream_t stream)
{
    if (R <= 0 || keep_pages <= 0) return;
    device::build_window_page_view<<<1, 256, 0, stream>>>(
        src_indices, src_indptr, keep_pages, dst_indptr, dst_indices, R);
    CUDA_CHECK(cudaGetLastError());
}

void build_full_split_view(
    const std::uint32_t* src_indptr,
    const std::uint32_t* src_last_page_len,
    int splits,
    int page_size,
    std::uint32_t* dst_indptr,
    std::uint32_t* dst_indices,
    std::uint32_t* dst_last,
    const std::uint32_t* src_indices,
    cudaStream_t stream)
{
    if (splits <= 0 || page_size <= 0) return;
    device::build_full_split_view<<<1, 32, 0, stream>>>(
        src_indptr, src_last_page_len, splits, page_size,
        dst_indptr, dst_indices, dst_last, src_indices);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels::attn
