#pragma once

/// RMSNorm, the residual add, and the gated norm: launch shapes.
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

namespace pie::kernels::norm {

using std::uint32_t;

// rms_single_row (Rms / FinalRms / QNorm / KNorm): one threadgroup per row,
// row_size/N_READS threads (N_READS=4). The row index is the threadgroup-x
// position (gid), so multi-row norms (per-head Q/K) stack rows on grid.x.
//   * Rms/FinalRms: n_rows=1, row_size=hidden(1024) → grid=(256,1,1) tg=(256,1,1)
//   * QNorm: n_rows=n_q_heads(8), row_size=head_dim(256) → grid=(512,1,1) tg=(64,1,1)
//   * KNorm: n_rows=n_kv_heads(2), row_size=head_dim(256) → grid=(128,1,1) tg=(64,1,1)
inline void rms_dispatch(int row_size, int n_rows, Grid& g, Threadgroup& tg) {
    // Rounded UP: `rms_single_row` guards its own tail, but a truncating
    // thread count silently drops the last partial group of 4 for any row
    // width that is not a multiple of N_READS.
    // Capped at what Metal allows a threadgroup to be; see `rms_mb_dispatch`,
    // which this one must agree with exactly at N == 1.
    const uint32_t t = std::min<uint32_t>((uint32_t(row_size) + 3) / 4, 1024);  // N_READS = 4
    g  = Grid{t * uint32_t(n_rows), 1, 1};
    tg = Threadgroup{t, 1, 1};
}

// residual_add (AttnResid / LayerOut): Out = X + Residual, elementwise over hidden.
inline void residual_dispatch(int hidden, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(hidden), 1, 1};
    tg = Threadgroup{256, 1, 1};
}

// gated_rms (GatedRms -> golden gdn_core): one threadgroup per value-head, V_d lanes
// cooperatively reduce. grid=(V_d, V_h, 1), tg=(V_d, 1, 1). head=tgpos.y, lane=lid.
inline void gated_rms_dispatch(int v_heads, int v_dim, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(v_dim), uint32_t(v_heads), 1};
    tg = Threadgroup{uint32_t(v_dim), 1, 1};
}


// ── the multi-batch shapes ──────────────────────────────────────────────────
//
// Moved verbatim from `driver-metal`'s `decode_dispatch_mb.hpp`. Every one is a
// pure function of the geometry: the DECISIONS that file also held --
// `qmm_bn`, `qmm_bn_unsplit`, `sdpa_should_tile` -- read `DeviceTuning` and
// stayed behind, so anything they choose arrives here as an argument.

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

/// `vnorm_single_row`: one threadgroup per row, the row's width in threads,
/// four elements each -- the same shape `rms_single_row` uses.
inline void vnorm_dispatch(int rows, int axis, Grid& g, Threadgroup& tg) {
    constexpr int kNReads = 4;
    const int threads = (axis + kNReads - 1) / kNReads;
    g = Grid{uint32_t(threads) * uint32_t(rows > 0 ? rows : 1), 1, 1};
    tg = Threadgroup{uint32_t(threads), 1, 1};
}

}  // namespace pie::kernels::norm
