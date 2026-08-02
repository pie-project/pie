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

#include "device_tuning.hpp"

namespace pie::metal {

// affine_qmv_fast over N token rows (batched GEMV). tid.x = token row (0..N-1), tid.y = out-row
// block. grid threads = (32*N, out/4, 1) → N*(out/8) threadgroups, tg=(32,2,1). At N=1 this is
// exactly qmv_dispatch (the sealed M=1 fast path). out%8==0 holds for every qwen3.6 projection.
inline void qmv_mb_dispatch(int out_vec, int N, Grid& g, Threadgroup& tg) {
    // Rounded UP, for the reason `qmv_dispatch` gives.
    g  = Grid{32u * uint32_t(N), (uint32_t(out_vec) + 3u) / 4u, 1};
    tg = Threadgroup{32, 2, 1};
}

// Below this batch the GEMV is the faster kernel. The crossover is a property
// of the MACHINE and of whether the checkpoint's FFN is routed -- see
// `device_tuning.hpp`. It was measured on an M1 Max, where pie's per-step cost
// beats mlx-lm's at every batch up to 8 with the GEMV and only loses above it;
// an M2 Max and an M4 Pro both move the DENSE crossover down to 8 and an
// unrecognised device still gets the 12 this constant was.
//
// Passed in rather than asked for here, which is the whole point: the value is
// not known until there is a device to ask AND a geometry to ask about, and
// this header has neither.
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
// The row blocks the GEMM is instantiated for, narrowest first. This is a
// LIST rather than a narrow/wide pair because the argument above does not stop
// at 32: a 128-row prompt at BM=32 still unpacks every weight four times, and
// measured on an M1 Max the third rung is worth as much as the second was --
// llama-1B prefills 1236 tok/s at BM=16, 1616 at 32 and 1936 at 64.
// A fourth rung would need more accumulators per lane than BM=64/BN=32's
// sixteen, which is where the register file stops paying.
//
// The other three axes were swept at this rung and none of them moved, so the
// list is where the tuning is:
//   BK=64  is SLOWER (1647 vs 1817 on llama-1B) and illegal below gs=64 --
//          QuantizedBlockLoader asserts BCOLS <= group_size, so a gs=32
//          checkpoint cannot compile it at all.
//   WM=4   is SLOWER (1714). Splitting a 64-row block four ways puts each lane
//          back to sixteen accumulators, and it does not help: the 32 that
//          WM=2 spends are not what the kernel is short of.
//   BN=32  is slower than 64 at this rung, so `qmm_bn`'s argument survives.
inline constexpr int kQmmBMs[] = {16, 32, 64};
inline constexpr int kQmmBMCount = int(sizeof(kQmmBMs) / sizeof(kQmmBMs[0]));
inline constexpr int kQmmBM = kQmmBMs[0];
inline constexpr int kQmmBMWide = kQmmBMs[kQmmBMCount - 1];

/// Which row block a batch of `N` rows should use, as an index into `kQmmBMs`.
/// A block only pays for itself once the batch can fill it, so the rule is the
/// widest rung the batch can cover. Callers pad the grid up to that rung, which
/// is why the row count they pad to must be asked of THIS function and not of
/// `kQmmBMWide`: padding a one-row decode to the widest block would launch 64
/// rows of arithmetic to compute one.
inline int qmm_bm_index(int N) {
    int best = 0;
    for (int i = 1; i < kQmmBMCount; ++i)
        if (N >= kQmmBMs[i]) best = i;
    return best;
}

inline int qmm_bm(int N) { return kQmmBMs[qmm_bm_index(N)]; }

/// The reverse: which `kQmmBMs` rung a chosen row block is. Dispatch records
/// the block it launched, not the batch it came from, so the PSO lookup has to
/// come back the other way.
inline int qmm_bm_slot(int bm) {
    for (int i = 0; i < kQmmBMCount; ++i)
        if (kQmmBMs[i] == bm) return i;
    return 0;
}

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

inline int qmm_bn(int out_vec, int N, int min_batch) {
    const int bm = qmm_bm(N);
    if (N < min_batch || N % bm != 0) return 0;
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

/// The threadgroup count past which this GEMM stops caring about more of them.
///
/// Where BN=32 starts beating BN=16, in threadgroups.
///
/// NOT the machine's saturation point, which is what an earlier version of this
/// called it. The sweep below brackets the crossover between 144 threadgroups
/// (where 16 still wins) and 192 (where 32 does); the machine saturates higher
/// than that, and using the saturation number here cost up to 12% because it
/// let the choice run past 32 to 64.
///
/// Read from `DeviceTuning`, not a constant: it is the threadgroup count at
/// which a wider tile stops being worth fewer of them, and how many
/// threadgroups fill the machine is the machine's business. The value above is
/// the M1 Max's; the M4 Pro re-measurement, which lands lower for exactly that
/// reason, is in `device_tuning.hpp`.
inline int qmm_bn_crossover_tg_value() { return qmm_bn_crossover_tg(); }

/// `qmm_bn` for a family whose GEMM has no split-K behind it.
///
/// The rule above is correct *because* the split supplies threadgroups when the
/// output tiles do not. A family that dispatches no split has no such supply,
/// and taking the widest tile then starves the machine: at M=128 with BM=64 a
/// projection to 1024 columns gets `1024/64 * 128/64` = 32 threadgroups, which
/// the curve prices at a third of what the same work does at 200.
///
/// So: the narrow tile until there is enough work to fill the machine, and 32
/// after that. Never 64 -- that is the finding, not an omission. Measured with
/// `roofline_probe` on gemma-4-E2B's own projections, BM=64, GFLOP/s, best of
/// each row starred:
///
///     M     N        tg@32    BN=16     BN=32     BN=64
///     128    512        16   *2187*     1829      1054
///     128   1024        32   *3249*     3115      2234
///     128   2048        64   *3399*     3346      3333
///     128   3584       112    3595     *3904*     3098
///     128   6144       192    3829     *4302*     4012
///     192   1536       144   *3820*     3529      2386
///     192   2048       192    3694     *4082*     3103
///     192   6144       576    3919     *4457*     4162
///     448    256        56   *2727*     2603      1896
///     448   1536       336    3865     *4101*     3573
///     448   2048       448    3820     *4337*     3851
///     448   6144      1344    3986     *4569*     4275
///    1024   1536       768    4000     *4565*     4252
///    1024   2048      1024    3941     *4511*     4131
///    1024   6144      3072    4016     *4619*     4326
///
/// Sixteen columns of measurement and BN=64 is the best of none of them, which
/// is why the rule no longer reaches for it. The threshold sits in the only gap
/// the sweep leaves: 144 threadgroups still wants 16, 192 already wants 32.
///
/// Checked against the machine and not just the probe, because the MIXTURE's
/// tile rule was built the same way and the probe misled it three times over.
/// Forcing each width through a real llama-3.2-1B prefill agrees with the
/// table: at 448 rows BN=16/32/64 give 2565.8 / 2663.7 / 2578.3 tok/s and at
/// 1024 rows 2270.8 / 2349.8 / 2297.0, so the rule's 32 is the machine's 32.
/// The dense side's probe holds where the mixture's did not, and the reason is
/// the mixture's alone: its threadgroups read thirty-two experts' weights
/// where the probe reads one.
///
/// BN partitions output columns only, so this is bit-exact whichever way it
/// goes; it decides how many times a weight tile is dequantized, not what the
/// sum is.
inline int qmm_bn_unsplit(int out_vec, int N, int min_batch) {
    const int bm = qmm_bm(N);
    if (N < min_batch || N % bm != 0 || out_vec % 16 != 0) return 0;
    const int row_tiles = N / bm;
    if (out_vec % 32 == 0 && (out_vec / 32) * row_tiles >= qmm_bn_crossover_tg_value())
        return 32;
    return 16;
}

// Split the K dimension when the output tiles alone leave the machine short.
// MLX picks the split to land near 512 threadgroups (backend/metal/
// quantized.cpp:880) and sends every transposed non-batched decode down this
// path rather than the plain GEMM; `roofline_probe` finds the same saturation
// point independently.  A projection to hidden (N=1024, 32 tiles) takes a split
// of 16, gate/up (N=3584, 112 tiles) takes 4, and lm_head has 7760 tiles of its
// own and takes none.
// 512 is MLX's number and it is this machine's too. An earlier sweep here
// preferred 256 and was measured against qwen3.5's split path, which never
// dispatched its reduce -- it was timing a kernel that computed the wrong
// answer, so the curve it drew was not this GEMM's curve. Re-swept on llama-1B
// at 32 lanes with the reduce in place: 741 tok/s at 128, 873 at 256, 887 at
// 512, 886 at 1024, 876 at 2048. Flat from 512 on, so take its near edge.
inline constexpr int kQmmSplitTargetTgs = 512;
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
    // Count the tiles the SPLIT dispatch will actually launch: `kQmmSplitBN`
    // wide and `bm` tall, which is the grid `qmm_t_splitk_dispatch` builds.
    //
    // This used to count rows in units of `kQmmBM`, the NARROWEST block, on the
    // theory that a wide block is twice as parallel and should be split half as
    // deep. That is backwards -- a wide block covers twice the rows in ONE
    // threadgroup, so it produces half the tiles and needs MORE split, not less
    // -- and the numbers that appeared to support it were measured on qwen3.5's
    // split path, which never dispatched its reduce and so was timing the
    // wrong answer. Counting honestly is worth 741 -> 870 tok/s at 32 lanes on
    // llama-1B.
    const int tiles = (out_vec / kQmmSplitBN) * ((N + bm - 1) / bm);
    static const int target = [] {
        const char* e = std::getenv("PIE_METAL_SPLIT_TGS");
        return e ? std::atoi(e) : kQmmSplitTargetTgs;
    }();
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
    static const bool off = std::getenv("PIE_METAL_NO_PREFILL_BM32") != nullptr;
    // The strided form is instantiated as a narrow/wide PAIR, not over
    // `kQmmBMs`, so it stops at 32 whatever the aligned table grew to.
    return (!off && padded_rows >= 32) ? 32 : kQmmBM;
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
    // Rounded up, matching `rms_dispatch`: at N == 1 these two must agree
    // exactly, because a family that uses this one for both is relying on it.
    //
    // Capped, because a threadgroup is not allowed to be any size: 1024 is what
    // Metal permits and `ceil(row_size / 4)` passes it at a hidden of 4100.
    // Nothing rejected the oversized ask -- the dispatch was simply not made
    // and the rows came out untouched, which is what Qwen3.6-27B (5120) and
    // gemma-4-31b (5376) were reading when they answered nonsense. The kernel
    // strides the row now, so a capped threadgroup still covers all of it.
    const uint32_t t = std::min<uint32_t>((uint32_t(row_size) + 3) / 4, 1024);  // N_READS = 4
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

// Query rows per threadgroup in `sdpa_paged_tiled` -- one per simdgroup, and a
// threadgroup is 1024 threads. It is the factor by which that kernel divides
// the K/V traffic, and it must equal the kernel's own QT.
inline constexpr int kSdpaQueryTile = 32;

// Whether to tile the query rows. The tiled kernel gives a row a simdgroup
// where the per-row kernel gives it a threadgroup, so below a full tile it is
// strictly worse: at one row it would run one simdgroup of the thirty-two the
// other kernel would have used. A fire earns the tiled shape by filling a tile.
/// Whether a fire's attention should use the tiled kernel rather than the
/// per-row one.
///
/// NOT a row count. The tiled kernel walks RUNS OF EQUAL REQUEST inside each
/// 32-row tile, staging that run's keys into threadgroup memory and letting
/// only the run's own simdgroups read them -- so its whole advantage is rows
/// that share a key span, and its cost is a serial pass per run. A prefill is
/// all one run and wins outright. A fleet of decodes is the opposite shape:
/// thirty-two rows, thirty-two runs, one simdgroup live per pass and the tile
/// staged thirty-two times. Measured on llama-1B, batch 32, that is 370 tok/s
/// tiled against 728 per-row, and batch 64 is 480 against 915.
///
/// Asking only `N >= kSdpaQueryTile` could not tell those apart, because the
/// two fires have the SAME row count. The request count is what separates
/// them, and the caller already has it: `qo_indptr` is one entry per request
/// plus a terminator.
inline bool sdpa_should_tile(int rows, int requests) {
    const int r = requests > 0 ? requests : 1;
    // Not `kSdpaQueryTile`, though it is the same number on this machine. That
    // one is the tile's HEIGHT and is the simdgroup count; this is a crossover
    // and belongs to the machine. See `DeviceTuning`.
    return rows / r >= sdpa_tile_min_rows_per_request();
}

// sdpa_paged_tiled: one threadgroup per (q_head, tile of kSdpaQueryTile rows).
// grid=(n_q_heads*1024, ceil(N/QT), 1), tg=(1024,1,1). The grid rounds UP, so
// the kernel reads N from bind::SdpaPaged::Rows to retire its partial tile.
inline void sdpa_paged_tiled_dispatch(int n_q_heads, int N, Grid& g, Threadgroup& tg) {
    const uint32_t tiles = uint32_t((N + kSdpaQueryTile - 1) / kSdpaQueryTile);
    g  = Grid{uint32_t(n_q_heads) * 1024u, tiles < 1u ? 1u : tiles, 1};
    tg = Threadgroup{1024, 1, 1};
}

// kv_append (paged, delta's kernel): one thread per (channel, kv_head, token). grid=
// (head_dim, n_kv_heads, N). Token m scatters to its phys_slot(position_ids[m]).
inline void kv_append_mb_dispatch(int head_dim, int n_kv_heads, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(head_dim), uint32_t(n_kv_heads), uint32_t(N)};
    tg = Threadgroup{uint32_t(head_dim), 1, 1};
}

}  // namespace pie::metal
