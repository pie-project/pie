#pragma once

// The three argmax launchers the ahead-of-time archive still needs.
//
// # What is NOT here any more, and why the list is short
//
// This header used to declare ten: `argmax_bf16`, `argmax_fp32`,
// `argmax_bf16_compact_scatter`, `lm_head_gemv_argmax_bf16`,
// `masked_embedding_argmax_bf16`, `topk_centroids_bf16` and
// `masked_embedding_tile_argmax_pairs_bf16` among them. `new-horizon.md`
// §41's transitive audit measured every one of them as reachable from no
// root at all — no `pie_k_*` shim entry, no sibling `.cu`, no test — and §43
// deleted them with their launchers. The JOBS did not go: they are device
// rows in `kernels_cuda_new::families::sample`, NVRTC compiles them out of
// `sample/argmax.cuh`, and `examples/unit_probe_*.rs` fires them there. A
// device row consumes the header and never the launcher, which is exactly
// why a launcher can be dead while its kernel is not.
//
// What remains is the half that has a consumer:
//
//   * `lm_head_gemv_argmax_int8` — `table::sample::KERNELS`' one row. Its
//     grid comes from an occupancy query, which is the case a `LaunchRule`
//     must not be invented for.
//   * `argmax_accumulate_bf16` / `argmax_finalize_bf16` — no row, and held
//     by `gemm/gemm.cpp`'s chunked LM-head argmax, which calls both. §10.10:
//     a launcher goes when its WHOLE consumer set has gone, and a sibling
//     translation unit is a consumer even when the row is not.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::sample {

// ── Chunked (vocab-streaming) argmax ─────────────────────────────────
// The sampler's read of a materialised [rows, vocab] logits tensor is pure
// HBM traffic that exists only because the reduction is a separate kernel
// from the LM head GEMM. When the caller can produce the logits one vocab
// slab at a time into a reused buffer, the slab stays L2-resident and both
// the write and the read stop reaching HBM (§20.36: 185 us/step at
// rows=512, vocab=151936).
//
// `argmax_accumulate_bf16` folds one slab into a running per-warp
// best; `argmax_finalize_bf16` collapses that to token ids. Slab order and
// scan order do not matter: the ordering is a total order on (value, -index),
// so the result is the argmax over the concatenated slabs.
//
// `acc_val` / `acc_idx` are caller-owned scratch of
// `rows * kArgmaxAccumSlots` elements each.
constexpr int kArgmaxAccumSlots = 32;

void argmax_accumulate_bf16(
    const void* slab,          // [rows, row_stride] bf16
    int rows,
    int width,                 // valid columns in this slab
    int row_stride,            // elements between slab rows
    int vocab_base,            // global vocab index of column 0
    float* acc_val,            // [rows, kArgmaxAccumSlots]
    std::int32_t* acc_idx,     // [rows, kArgmaxAccumSlots]
    bool init,                 // true for the first slab
    cudaStream_t stream);

void argmax_finalize_bf16(
    const float* acc_val,
    const std::int32_t* acc_idx,
    std::int32_t* token_ids,   // [rows]
    int rows,
    cudaStream_t stream);

// Fused GEMV + argmax for MTP lm_head scoring. Returns greedy token IDs
// without materializing logits.
void lm_head_gemv_argmax_int8(
    const void* hidden_states,        // [num_rows, hidden] bf16
    const std::int8_t* lm_head_weight, // [vocab, hidden] int8
    const float* scale_inv,           // [vocab] fp32 per-channel
    std::int32_t* token_ids,          // [num_rows]
    int num_rows,
    int hidden,
    int vocab,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::sample
