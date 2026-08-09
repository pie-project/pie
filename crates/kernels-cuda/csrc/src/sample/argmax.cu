// The launchers, and nothing else.
//
// Every `__global__` this file used to hold now lives in `sample/argmax.cuh`,
// which the JIT compiles at run time and which this file includes so the
// ahead-of-time archive keeps exactly ONE definition of each. What stays here
// is the half NVRTC cannot have, and in this family that half is large:
//
//   * `argmax_vec2_usable` -- a run-time test on a pointer's alignment and
//     the vocab's parity that decides WHICH kernel to fire.
//   * `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`, which is
//     where the fused GEMV's grid comes from.
//   * A `static` scratch buffer that `cudaMalloc`s and grows, and launchers
//     that fire two kernels to use it.
//   * `centroid_top_k` clamped to the shared-memory shortlist the masked
//     kernels index with it.
//
// Three kernels have rows in `kernels_cuda_new::families::sample`; the other
// ten are blocked on one of the four above, and `sample/argmax.cuh`'s header
// says which for each. A launcher whose grid comes from an occupancy query is
// exactly the case a `LaunchRule` must not be invented for.
//
// The scalar layer and the fixed-width integer names come out of the prelude,
// through the device header: NVRTC has no CUDA device headers, and
// `sample/argmax.cuh` is meant to compile under both it and nvcc.
#include "pie_device.cuh"
#include "sample/argmax.cuh"
#include "sample/argmax.hpp"

#include <cstddef>
#include <cstdlib>

namespace pie_cuda_driver::kernels::sample {

// The accumulator carries one slot per warp, and the header publishes that
// count to its callers as `kArgmaxAccumSlots`. The kernel's width and the
// header's promise are two files, so the agreement is asserted rather than
// assumed.
static_assert(device::kAccumWarps == kArgmaxAccumSlots,
              "the accumulator carries one slot per warp; "
              "kArgmaxAccumSlots must match kAccumThreads / 32");

namespace {

// The vec2 kernels index rows as `base + row * vocab` and load through
// `device::bf16x2`, so an odd vocab puts every second row on a 2-byte
// boundary and the load faults. Production vocabs are even, which is why this
// never fired, but the guard costs nothing.
bool argmax_vec2_usable(const void* logits, int vocab) {
    return (vocab % 2) == 0 &&
           (reinterpret_cast<std::uintptr_t>(logits) % 4) == 0;
}

}  // namespace

void argmax_bf16(
    const void* logits, device::i32* token_ids,
    int num_rows, int vocab, cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(device::BLOCK);
    if (argmax_vec2_usable(logits, vocab)) {
        device::argmax_vec2<device::bf16><<<grid, block, 0, stream>>>(
            static_cast<const device::bf16*>(logits), token_ids, vocab);
    } else {
        device::argmax<device::bf16><<<grid, block, 0, stream>>>(
            static_cast<const device::bf16*>(logits), token_ids, vocab);
    }
}

void argmax_bf16_compact_scatter(
    const void* logits,
    const device::i32* row_indices,
    device::i32* token_ids,
    int num_rows,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || vocab <= 0) return;
    dim3 grid(num_rows);
    dim3 block(device::BLOCK);
    if (argmax_vec2_usable(logits, vocab)) {
        device::argmax_compact_scatter_vec2<device::bf16><<<grid, block, 0, stream>>>(
            static_cast<const device::bf16*>(logits), row_indices,
            token_ids, vocab);
    } else {
        device::argmax_compact_scatter<device::bf16><<<grid, block, 0, stream>>>(
            static_cast<const device::bf16*>(logits), row_indices,
            token_ids, vocab);
    }
}

void argmax_accumulate_bf16(
    const void* slab,
    int rows,
    int width,
    int row_stride,
    int vocab_base,
    float* acc_val,
    device::i32* acc_idx,
    bool init,
    cudaStream_t stream)
{
    if (rows <= 0 || width <= 0) return;
    // The vectorised path needs every row to start 16-byte aligned, which is
    // the slab base and the row stride together. A caller slicing a chunk out
    // of a wider buffer can land on an odd column, so check rather than assume.
    const bool vectorised =
        (row_stride % 8) == 0 && width >= 8 &&
        (reinterpret_cast<std::uintptr_t>(slab) % 16) == 0;
    device::argmax_accumulate<device::bf16>
        <<<rows, device::kAccumThreads, 0, stream>>>(
            static_cast<const device::bf16*>(slab), width, row_stride,
            vocab_base, acc_val, acc_idx, init, vectorised);
}

void argmax_finalize_bf16(
    const float* acc_val,
    const device::i32* acc_idx,
    device::i32* token_ids,
    int rows,
    cudaStream_t stream)
{
    if (rows <= 0) return;
    device::argmax_finalize<<<rows, device::kAccumWarps, 0, stream>>>(
        acc_val, acc_idx, token_ids);
}

void lm_head_gemv_argmax_int8(
    const void* hidden_states,
    const device::i8* lm_head_weight,
    const float* scale_inv,
    device::i32* token_ids,
    int num_rows,
    int hidden,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || vocab <= 0) return;

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    constexpr int kBlocksPerSm = 2;
    const int max_blocks_x = num_sms * kBlocksPerSm;
    const int min_blocks_x =
        (vocab + device::GEMV_WARPS - 1) / device::GEMV_WARPS;
    const int num_blocks_x = std::min(max_blocks_x, min_blocks_x);
    const device::usize shmem_bytes =
        static_cast<device::usize>(hidden) * sizeof(float);

    const device::usize pairs_elems =
        static_cast<device::usize>(num_blocks_x) * num_rows;
    static device::u64* s_partial_pairs = nullptr;
    static device::usize s_pairs_cap = 0;
    if (pairs_elems > s_pairs_cap) {
        if (s_partial_pairs) cudaFree(s_partial_pairs);
        cudaMalloc(&s_partial_pairs, pairs_elems * sizeof(device::u64));
        s_pairs_cap = pairs_elems;
    }

    dim3 grid(num_blocks_x, num_rows);
    dim3 block(device::GEMV_BLOCK_DIM);
    device::lm_head_gemv_argmax_int8<device::bf16>
        <<<grid, block, shmem_bytes, stream>>>(
            static_cast<const device::bf16*>(hidden_states),
            lm_head_weight,
            scale_inv,
            s_partial_pairs,
            num_rows,
            hidden,
            vocab,
            num_blocks_x);

    dim3 sel_block(128);
    dim3 sel_grid((num_rows + sel_block.x - 1) / sel_block.x);
    device::select_lm_head_argmax_pairs<<<sel_grid, sel_block, 0, stream>>>(
        s_partial_pairs, token_ids, num_rows, num_blocks_x);
}

void lm_head_gemv_argmax_bf16(
    const void* hidden_states,
    const void* lm_head_weight,
    device::i32* token_ids,
    int num_rows,
    int hidden,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || vocab <= 0) return;

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int blocks_per_sm = 4;
    const int max_blocks_x = num_sms * blocks_per_sm;
    const int min_blocks_x =
        (vocab + device::GEMV_WARPS - 1) / device::GEMV_WARPS;
    const int num_blocks_x = std::min(max_blocks_x, min_blocks_x);
    const device::usize shmem_bytes =
        static_cast<device::usize>(hidden) * sizeof(float);

    const device::usize pairs_elems =
        static_cast<device::usize>(num_blocks_x) * num_rows;
    static device::u64* s_partial_pairs_bf16 = nullptr;
    static device::usize s_pairs_cap_bf16 = 0;
    if (pairs_elems > s_pairs_cap_bf16) {
        if (s_partial_pairs_bf16) cudaFree(s_partial_pairs_bf16);
        cudaMalloc(&s_partial_pairs_bf16, pairs_elems * sizeof(device::u64));
        s_pairs_cap_bf16 = pairs_elems;
    }

    dim3 grid(num_blocks_x, num_rows);
    dim3 block(device::GEMV_BLOCK_DIM);
    device::lm_head_gemv_argmax<device::bf16>
        <<<grid, block, shmem_bytes, stream>>>(
            static_cast<const device::bf16*>(hidden_states),
            static_cast<const device::bf16*>(lm_head_weight),
            s_partial_pairs_bf16,
            num_rows,
            hidden,
            vocab,
            num_blocks_x);

    dim3 sel_block(128);
    dim3 sel_grid((num_rows + sel_block.x - 1) / sel_block.x);
    device::select_lm_head_argmax_pairs<<<sel_grid, sel_block, 0, stream>>>(
        s_partial_pairs_bf16, token_ids, num_rows, num_blocks_x);
}

void argmax_fp32(
    const void* logits,
    device::i32* token_ids,
    int num_rows,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || vocab <= 0) return;
    dim3 grid(num_rows);
    dim3 block(device::BLOCK);
    device::argmax<device::f32><<<grid, block, 0, stream>>>(
        static_cast<const float*>(logits), token_ids, vocab);
}

void masked_embedding_argmax_bf16(
    const void* centroid_logits,
    const void* hidden_states,
    const void* lm_head_weight,
    const device::i64* token_ordering,
    device::i32* token_ids,
    int num_rows,
    int hidden,
    int num_centroids,
    int centroid_top_k,
    int vocab_per_centroid,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || num_centroids <= 0 ||
        centroid_top_k <= 0 || vocab_per_centroid <= 0) {
        return;
    }
    if (centroid_top_k > device::MAX_MASKED_TOP_K) {
        centroid_top_k = device::MAX_MASKED_TOP_K;
    }
    dim3 grid(num_rows);
    dim3 block(device::BLOCK);
    device::masked_embedding_argmax<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(centroid_logits),
        static_cast<const device::bf16*>(hidden_states),
        static_cast<const device::bf16*>(lm_head_weight),
        token_ordering,
        token_ids,
        hidden,
        num_centroids,
        centroid_top_k,
        vocab_per_centroid);
}

void topk_centroids_bf16(
    const void* centroid_logits,
    device::i32* top_centroids,
    int num_rows,
    int num_centroids,
    int centroid_top_k,
    cudaStream_t stream)
{
    if (num_rows <= 0 || num_centroids <= 0 || centroid_top_k <= 0) return;
    if (centroid_top_k > device::MAX_MASKED_TOP_K) {
        centroid_top_k = device::MAX_MASKED_TOP_K;
    }
    dim3 grid(num_rows);
    dim3 block(device::BLOCK);
    device::topk_centroids<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(centroid_logits),
        top_centroids,
        num_centroids,
        centroid_top_k);
}

void masked_embedding_tile_argmax_pairs_bf16(
    const device::i32* top_centroids,
    const void* hidden_states,
    const void* lm_head_weight,
    const device::i64* token_ordering,
    device::u64* partial_pairs,
    int num_rows,
    int hidden,
    int centroid_top_k,
    int vocab_per_centroid,
    int num_tiles,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || centroid_top_k <= 0 ||
        vocab_per_centroid <= 0 || num_tiles <= 0) {
        return;
    }
    if (centroid_top_k > device::MAX_MASKED_TOP_K) {
        centroid_top_k = device::MAX_MASKED_TOP_K;
    }
    const int selected = centroid_top_k * vocab_per_centroid;
    dim3 grid(num_rows, num_tiles);
    dim3 block(device::BLOCK);
    device::masked_embedding_tile_argmax_pairs<device::bf16>
        <<<grid, block, 0, stream>>>(
            top_centroids,
            static_cast<const device::bf16*>(hidden_states),
            static_cast<const device::bf16*>(lm_head_weight),
            token_ordering,
            partial_pairs,
            hidden,
            centroid_top_k,
            vocab_per_centroid,
            selected,
            num_tiles);
}

}  // namespace pie_cuda_driver::kernels::sample
