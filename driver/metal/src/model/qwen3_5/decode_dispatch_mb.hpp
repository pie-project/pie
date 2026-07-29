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
#include "decode_dispatch.hpp"  // M=1 helpers (qmv_dispatch, rms_dispatch, ...)
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
inline constexpr int kQmmMinThreadgroups = 192;

inline int qmm_bn(int out_vec, int N) {
    const int bm = qmm_bm(N);
    if (N < kQmmMinBatch || N % bm != 0) return 0;
    const int row_blocks = N / bm;
    int best = 0;
    for (int bn : {16, 32, 64}) {
        if (out_vec % bn != 0) continue;
        if (best == 0) best = bn;  // the narrowest that divides, as a floor
        if ((out_vec / bn) * row_blocks >= kQmmMinThreadgroups) best = bn;
    }
    return best;
}

// `out/BN` threadgroups across the output, `M/BM` across the batch, each
// 32x2x2 = 128 threads (WM=WN=2 simdgroups), which is the shape steel's
// BlockMMA is written for.
// The prefill's batched projection. Rows are padded up to a whole BM tile: the
// scratch pool holds `max_tokens` rows and the tail rows land in ones the fire
// does not use, so the padding computes discardable values rather than needing
// a bounds-checked inner loop.
inline int qmm_strided_rows(int N, int max_rows) {
    const int padded = ((N + kQmmBM - 1) / kQmmBM) * kQmmBM;
    return padded <= max_rows ? padded : 0;
}

inline void qmm_t_strided_dispatch(int out_vec, int padded_rows, Grid& g,
                                   Threadgroup& tg) {
    g  = Grid{32u * (uint32_t(out_vec) / 32u),
              2u * (uint32_t(padded_rows) / uint32_t(kQmmBM)), 2};
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
