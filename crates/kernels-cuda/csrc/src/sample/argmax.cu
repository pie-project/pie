// The launchers, and nothing else.
//
// Every `__global__` this file used to hold now lives in `sample/argmax.cuh`,
// which the JIT compiles at run time and which this file includes so the
// ahead-of-time archive keeps exactly ONE definition of each. What stays here
// is the half NVRTC cannot have, and after §43's sweep that half is three
// launchers rather than ten:
//
//   * `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`, which is
//     where the fused GEMV's grid comes from.
//   * A `static` scratch buffer that `cudaMalloc`s and grows, and a launcher
//     that fires two kernels to use it.
//
// One of the three is a row (`table::sample`'s `lm_head_gemv_argmax_int8`)
// and two are held by a sibling `.cpp`. Everything else this file used to
// declare was reachable from nothing at all -- see the note below the
// `static_assert`. A launcher whose grid comes from an occupancy query is
// exactly the case a `LaunchRule` must not be invented for, and that is why
// the one that stays, stays.
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

// # Seven launchers went from here, and one predicate with them
//
// `argmax_bf16`, `argmax_bf16_compact_scatter`, `argmax_fp32`,
// `lm_head_gemv_argmax_bf16`, `masked_embedding_argmax_bf16`,
// `topk_centroids_bf16` and `masked_embedding_tile_argmax_pairs_bf16` were
// host launchers no root could reach: no `pie_k_*` shim entry forwards to
// them, no `.cu` calls them, and `table::sample::KERNELS` holds exactly one
// row -- `lm_head_gemv_argmax_int8`, still below. What names the other jobs
// is `families::sample`, whose rows are DEVICE rows: NVRTC compiles
// `sample/argmax.cuh` and the `examples/unit_probe_*.rs` probes fire them
// there, which is a consumer of the header and never of a launcher here.
//
// `argmax_vec2_usable` went with them because it was their predicate and
// only theirs -- the vec2/scalar choice is stated on the device rows as a
// `Select`, which is where it belongs. The remaining launchers make no such
// choice, so the file no longer needs the guard. If a vec2 launcher comes
// back, the predicate comes back with it, from `families::sample`'s terms
// rather than from a copy here.
//
// `argmax_accumulate_bf16` and `argmax_finalize_bf16` STAY and are not
// rows: `gemm/gemm.cpp`'s chunked LM-head argmax calls both. §10.10 -- a
// launcher goes when its WHOLE consumer set has gone, and a sibling `.cu`
// is a consumer even when the row is not.


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






}  // namespace pie_cuda_driver::kernels::sample
