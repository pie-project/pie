#pragma once

/// The quantised projections' launch shapes.
///
/// Read off the kernels' `[[thread_position_in_grid]]` contracts, which is why
/// they are here and not with the model that dispatches them --
/// `driver-metal`'s `shared_kernels.hpp` states the same rule for buffer
/// layouts one crate up. Moved verbatim from
/// `driver-metal/csrc/src/model/qwen3_5/decode_dispatch.hpp`, which kept them
/// only because that is where the first family to need them was written.
///
/// `dispatchThreads` semantics throughout: `Grid` is TOTAL THREADS and `tg` is
/// threads per threadgroup, so threadgroups are `ceil(Grid / tg)` per axis.

#include <algorithm>
#include <cstdint>

#include "pie/kernels/grid.h"

namespace pie::kernels::quant {

using std::uint32_t;

// affine_qmv_fast (all Qmv* kinds): tg=(32,2,1); grid threads=(32, N/4, 1)
// → N/8 threadgroups. Requires N%8==0 (holds for every qwen3.6 projection,
// incl. lm_head N=vocab=248320). K is a bound constant, not a launch dim.
inline void qmv_dispatch(int N, Grid& g, Threadgroup& tg) {
    // Rounded UP. `affine_qmv_fast` produces four outputs per simdgroup, so a
    // truncating count drops every output past the last whole four -- and at
    // N < 4 it drops the dispatch entirely. The shared expert's gate is
    // hidden -> ONE logit a token: its grid was {32, 0, 1}, no threads ran,
    // its buffer kept the zeros it was allocated with, and every routed token
    // was combined under `sigmoid(0) = 0.5` instead of its own gate.
    // The 4 is `results_per_simdgroup` in `quantized_qmv.metal`, which was swept
    // and is at a peak in both directions -- see the table there before moving it.
    g  = Grid{32, (uint32_t(N) + 3u) / 4u, 1};
    tg = Threadgroup{32, 2, 1};
}


// ── the multi-batch shapes ──────────────────────────────────────────────────
//
// Moved verbatim from `driver-metal`'s `decode_dispatch_mb.hpp`. Every one is a
// pure function of the geometry: the DECISIONS that file also held --
// `qmm_bn`, `qmm_bn_unsplit`, `sdpa_should_tile` -- read `DeviceTuning` and
// stayed behind, so anything they choose arrives here as an argument.

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

// affine_qmv_fast over N token rows (batched GEMV). tid.x = token row (0..N-1), tid.y = out-row
// block. grid threads = (32*N, out/4, 1) → N*(out/8) threadgroups, tg=(32,2,1). At N=1 this is
// exactly qmv_dispatch (the sealed M=1 fast path). out%8==0 holds for every qwen3.6 projection.
inline void qmv_mb_dispatch(int out_vec, int N, Grid& g, Threadgroup& tg) {
    // Rounded UP, for the reason `qmv_dispatch` gives.
    // The 4 is `results_per_simdgroup` in `quantized_qmv.metal`, which was swept
    // and is at a peak in both directions -- see the table there before moving it.
    g  = Grid{32u * uint32_t(N), (uint32_t(out_vec) + 3u) / 4u, 1};
    tg = Threadgroup{32, 2, 1};
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

/// `bm` is passed rather than recomputed: the encoder may have had to narrow it
/// to a rung whose pipeline exists, and the grid is the ONLY thing that tells
/// this kernel how many rows it has.
inline void qmm_t_strided_dispatch(int out_vec, int padded_rows, int bm, Grid& g,
                                   Threadgroup& tg) {
    g  = Grid{32u * (uint32_t(out_vec) / 32u),
              2u * (uint32_t(padded_rows) / uint32_t(bm > 0 ? bm : kQmmBM)), 2};
    tg = Threadgroup{32, 2, 2};
}

inline void qmm_t_dispatch(int out_vec, int N, int bn, int bm, Grid& g, Threadgroup& tg) {
    g  = Grid{32u * (uint32_t(out_vec) / uint32_t(bn)),
              2u * (uint32_t(N) / uint32_t(bm)), 2};
    tg = Threadgroup{32, 2, 2};
}

}  // namespace pie::kernels::quant
