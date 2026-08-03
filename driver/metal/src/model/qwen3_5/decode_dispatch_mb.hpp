#pragma once
// decode_dispatch_mb.hpp — beta's M>1 launch-geometry helpers (multi-batch lane).
//
// The N>1 generalization of decode_dispatch.hpp. Pie's batch dim is N=total_tokens; the
// raw-Metal activations are token-major [N, feature], so every per-row kernel just stacks N
// rows on the batch axis. KEY INSIGHT (quantized_qmv.metal): affine_qmv_fast ALREADY batches
// over tid.x (`x += tid.x*in_vec_size; y += tid.x*out_vec_size`) — the M=1 path launches with
// 1 threadgroup in x. So an M>1 batched GEMV is just `grid.x *= N`, BIT-EXACT by construction
// (each row reproduces the M=1 accumulation) and reducing to the shipped path at N=1. No new
// GEMM kernel is needed for CORRECTNESS; the tiled affine_qmm (weight reuse across rows) is a
// PERF lever layered on AFTER the parity gate is green.
//
// Pure (no Metal types beyond Grid/Threadgroup). dispatchThreads semantics: Grid = TOTAL
// THREADS, tg = threads/threadgroup, threadgroups = ceil(Grid/tg) per axis (matches
// decode_dispatch.hpp + RawMetalContext::dispatch).

#include "decode_abi.hpp"
#include <algorithm>
#include <cstdlib>

#include "decode_dispatch.hpp"
#include "decode_step_mb.hpp"  // M=1 helpers (qmv_dispatch, rms_dispatch, ...)
#include "mtl4_context.hpp"     // Grid, Threadgroup

namespace pie::metal {

// affine_qmv_fast over N token rows (batched GEMV). tid.x = token row (0..N-1), tid.y = out-row
// block. grid threads = (32*N, out/4, 1) → N*(out/8) threadgroups, tg=(32,2,1). At N=1 this is
// exactly qmv_dispatch (the sealed M=1 fast path). out%8==0 holds for every qwen3.6 projection.
inline void qmv_mb_dispatch(int out_vec, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{32u * uint32_t(N), uint32_t(out_vec) / 4, 1};
    tg = Threadgroup{32, 2, 1};
}

// Below this batch the GEMV is the faster kernel: measured, pie's per-step cost
// beats mlx-lm's at every batch up to 8 with the GEMV and only loses above it.
inline constexpr int kQmmMinBatch = 12;
// The ported steel GEMM is instantiated aligned-only, at BM=16 and BK=32. K is
// not checked: every qwen3.6 projection has K % 512 == 0 (the same fact the
// GEMV port relies on for its "fast" variant), so K % BK == 0 is free.
// Rows per threadgroup.  The GEMM dequantizes a weight tile once per row
// block, so a batch that spans several blocks pays for the same dequantize
// again in each -- which is why doubling M nearly doubles the time (14.6ms at
// M=16, 24.4 at 32, 45.7 at 64, measured standalone across the checkpoint's
// projections).  A taller block halves that work at the cost of halving the
// threadgroup count, so it is only worth taking once the batch is wide enough
// to have blocks to spare: at M=32, BM=32 measures 20.4ms against BM=16's 24.4.
inline constexpr int kQmmBM = 16;
inline constexpr int kQmmBMWide = 32;
inline constexpr int kQmmWideMinBatch = 32;

inline int qmm_bm(int N) { return N >= kQmmWideMinBatch ? kQmmBMWide : kQmmBM; }

// Output columns per threadgroup.  This GEMM is occupancy-bound, not bandwidth-
// bound: measured standalone at the model's shapes it turns in ~380 GFLOP/s at
// 16 threadgroups, ~900 at 32, ~1600 at 112 and saturates around 2.2-2.6 TFLOP/s
// past ~200, against a 389 GB/s machine it only ever reaches 11% of.  So take
// the widest tile -- wider means each x row is loaded fewer times -- that still
// leaves enough threadgroups to fill the machine, and past that prefer more
// threadgroups over a wider tile.
//
// The measured optimum for every projection in the checkpoint falls out of that
// one rule: BN=64 for lm_head (3880 tg), 32 for the GDN in-projection (192),
// 16 for everything else.  The old rule asked only whether `out_vec/64 >= 64`,
// which handed the GDN in-projection a BN=64 that measured 21% slower.
//
// BN partitions output columns only -- every element's K sum is unchanged -- so
// the choice is bit-exact whichever way it goes.

inline int qmm_bn(int out_vec, int N) {
    const int bm = qmm_bm(N);
    if (N < kQmmMinBatch || N % bm != 0) return 0;
    // Take the WIDEST tile that divides the output, full stop.
    //
    // This used to gate on a threadgroup count, and that was right when the
    // GEMM had nothing else supplying parallelism: measured then, BN=64 lost
    // everywhere except lm_head and the occupancy rule was worth 677 -> 712
    // tok/s.  Split-K changed the premise.  The split now supplies the
    // threadgroups, so the only thing BN still decides is how many times each
    // weight tile is dequantized -- and wider is strictly fewer.
    //
    // Interleaved A/B, decode step, widest against the old 192-threadgroup
    // rule: 16 lanes 31.57ms to 37.02, 32 lanes 141.18 to 158.45.  The old
    // rule is a pessimization now.
    int best = 0;
    for (int bn : {16, 32, 64})
        if (out_vec % bn == 0) best = bn;
    return best;
}

// Split the K dimension when the output tiles alone leave the machine short.
// MLX picks the split to land near 512 threadgroups (backend/metal/
// quantized.cpp:880) and sends every transposed non-batched decode down this
// path rather than the plain GEMM; `roofline_probe` finds the same saturation
// point independently.  A projection to hidden (N=1024, 32 tiles) takes a split
// of 16, gate/up (N=3584, 112 tiles) takes 4, and lm_head has 7760 tiles of its
// own and takes none.
// MLX targets 512 threadgroups here, which is right for the hardware it was
// tuned on and 7% wrong for this one: swept on an M1 Max, the decode step is
// 29.0ms at 64, 24.7 at 128, 24.7 at 256 and 26.1 at 512, and an interleaved
// A/B against MLX's value reads 86.85ms to 93.61ms at 32 lanes.  256 sits in
// the middle of the flat region rather than on its edge.
//
// The shape of the curve is the reason: past the point where the machine is
// full, more partitions only add reduce traffic, and a 32-core M1 Max fills at
// a lower count than the parts MLX tunes for.
inline constexpr int kQmmSplitTargetTgs = 256;
inline constexpr int kQmmSplitBN = 32;
inline constexpr int kQmmSplitMaxSplits = 16;
// The widest projection that takes this path.  lm_head has enough output tiles
// of its own to never need a split, which is what keeps the partials buffer to
// a few MB instead of the vocabulary's hundreds.
inline constexpr int kQmmSplitMaxOut = 8192;

// Each partition must be a whole number of BK-wide tiles AND whole quantization
// groups, or it reads into the next group's scales.
inline int qmm_split_k(int out_vec, int N, int K, int bm) {
    if (out_vec % kQmmSplitBN != 0 || bm <= 0) return 1;
    // Count the batch in units of the NARROW row block, not the one this
    // dispatch happens to use.  A wide block covers twice the rows in one
    // threadgroup, so counting by it would call a 32-row batch as parallel as a
    // 16-row one and split both the same -- measured at 32 lanes that costs
    // 8% (32.37ms split against 29.86 unsplit), where at 16 lanes splitting
    // wins 11% (18.29 against 20.50).
    const int tiles = (out_vec / kQmmSplitBN) * ((N + kQmmBM - 1) / kQmmBM);
    constexpr int target = kQmmSplitTargetTgs;
    int split = tiles > 0 ? target / tiles : 1;
    split = std::min(split, kQmmSplitMaxSplits);
    const int k_align = 64;  // group_size, and a multiple of BK=32
    split = std::min(split, K / k_align);
    while (split > 1 && K % (split * k_align) != 0) --split;
    return split < 2 ? 1 : split;
}

inline void qmm_t_splitk_dispatch(int out_vec, int N, int bm, int split, Grid& g,
                                  Threadgroup& tg) {
    // dispatchThreads: (tiles_n * 32 lanes, tiles_m * 2, split * 2).
    g  = Grid{32u * (uint32_t(out_vec) / uint32_t(kQmmSplitBN)),
              2u * uint32_t((N + bm - 1) / bm), 2u * uint32_t(split)};
    tg = Threadgroup{32, 2, 2};
}

inline void qmm_splitk_reduce_dispatch(int out_vec, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(out_vec), uint32_t(N), 1};
    tg = Threadgroup{256, 1, 1};
}

// `out/BN` threadgroups across the output, `M/BM` across the batch, each
// 32x2x2 = 128 threads (WM=WN=2 simdgroups), which is the shape steel's
// BlockMMA is written for.
// The prefill's batched projection. Rows are padded up to a whole BM tile: the
// scratch pool holds `max_tokens` rows and the tail rows land in ones the fire
// does not use, so the padding computes discardable values rather than needing
// a bounds-checked inner loop.
// A prompt has far more rows than a decode batch, so it can afford the wide
// row block -- and needs it for the same reason the decode does: the tile is
// dequantized once per row block, so a 512-row prompt at BM=16 unpacks every
// weight thirty-two times.
inline int qmm_strided_bm(int padded_rows) {
    return padded_rows >= kQmmWideMinBatch ? kQmmBMWide : kQmmBM;
}

inline int qmm_strided_rows(int N, int max_rows) {
    const int bm = qmm_strided_bm(N);
    const int padded = ((N + bm - 1) / bm) * bm;
    return padded <= max_rows ? padded : 0;
}

inline void qmm_t_strided_dispatch(int out_vec, int padded_rows, Grid& g,
                                   Threadgroup& tg) {
    g  = Grid{32u * (uint32_t(out_vec) / 32u),
              2u * (uint32_t(padded_rows) / uint32_t(qmm_strided_bm(padded_rows))), 2};
    tg = Threadgroup{32, 2, 2};
}

inline void qmm_t_dispatch(int out_vec, int N, int bn, int bm, Grid& g, Threadgroup& tg) {
    g  = Grid{32u * (uint32_t(out_vec) / uint32_t(bn)),
              2u * (uint32_t(N) / uint32_t(bm)), 2};
    tg = Threadgroup{32, 2, 2};
}

// rms_single_row over N tokens × n_rows rows-per-token (e.g. per-head q/k norm). One
// threadgroup per row; rows stack token-major [N*n_rows, row_size]. grid.x = (row_size/4)*n_rows*N.
inline void rms_mb_dispatch(int row_size, int n_rows, int N, Grid& g, Threadgroup& tg) {
    const uint32_t t = uint32_t(row_size) / 4;  // N_READS = 4
    g  = Grid{t * uint32_t(n_rows) * uint32_t(N), 1, 1};
    tg = Threadgroup{t, 1, 1};
}

// Elementwise over N rows × `width` channels (residual_add / silu_mul / attn_gate). Token-major
// [N, width]; one thread per (row, channel) folded onto grid.x. tg 256.
inline void elementwise_mb_dispatch(int width, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(width) * uint32_t(N), 1, 1};
    tg = Threadgroup{256, 1, 1};
}

// embed_gather_mb over N tokens: thread (channel k, token m). Token m gathers id[m].
// out token-major [N, hidden]. grid=(hidden, N, 1), tg=(256,1,1).
inline void embed_mb_dispatch(int hidden, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(hidden), uint32_t(N), 1};
    tg = Threadgroup{256, 1, 1};
}

// rope over N tokens: pos.x = freq idx, pos.y = head, pos.z = token row. Token m reads
// position_ids[m] (per-row IO read). grid=(rotary/2, n_heads, N), tg=(rotary/2,1,1).
inline void rope_mb_dispatch(int rotary_dims, int n_heads, int N, Grid& g, Threadgroup& tg) {
    const uint32_t half = uint32_t(rotary_dims) / 2;
    g  = Grid{half, uint32_t(n_heads), uint32_t(N)};
    tg = Threadgroup{half, 1, 1};
}

// sdpa_paged_decode: one threadgroup per (q_head, query row). grid=(n_q_heads*1024, N, 1),
// tg=(1024,1,1). Causal bound per row = position_ids[row]; request = req_of_token[row].
inline void sdpa_paged_dispatch(int n_q_heads, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(n_q_heads) * 1024u, uint32_t(N), 1};
    tg = Threadgroup{1024, 1, 1};
}

// kv_append (paged, delta's kernel): one thread per (channel, kv_head, token). grid=
// (head_dim, n_kv_heads, N). Token m scatters to its phys_slot(position_ids[m]).
inline void kv_append_mb_dispatch(int head_dim, int n_kv_heads, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(head_dim), uint32_t(n_kv_heads), uint32_t(N)};
    tg = Threadgroup{uint32_t(head_dim), 1, 1};
}

}  // namespace pie::metal
