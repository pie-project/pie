#pragma once

/// Rotary embedding: launch shapes.
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

namespace pie::kernels::rope {

using std::uint32_t;

// rope_neox_decode: pos.x = freq index (0..rotary_dims/2-1), pos.y = head.
// In-place single-tensor → dispatched once for Q (n_q_heads) and once for K
// (n_kv_heads). grid=(rotary_dims/2, n_heads, 1), tg=(rotary_dims/2, 1, 1).
inline void rope_dispatch(int rotary_dims, int n_heads, Grid& g, Threadgroup& tg) {
    const uint32_t half = uint32_t(rotary_dims) / 2;
    g  = Grid{half, uint32_t(n_heads), 1};
    tg = Threadgroup{half, 1, 1};
}


// ── the multi-batch shapes ──────────────────────────────────────────────────
//
// Moved verbatim from `driver-metal`'s `decode_dispatch_mb.hpp`. Every one is a
// pure function of the geometry: the DECISIONS that file also held --
// `qmm_bn`, `qmm_bn_unsplit`, `sdpa_should_tile` -- read `DeviceTuning` and
// stayed behind, so anything they choose arrives here as an argument.

// rope over N tokens: pos.x = freq idx, pos.y = head, pos.z = token row. Token m reads
// position_ids[m] (per-row IO read). grid=(rotary/2, n_heads, N), tg=(rotary/2,1,1).
inline void rope_mb_dispatch(int rotary_dims, int n_heads, int N, Grid& g, Threadgroup& tg) {
    const uint32_t half = uint32_t(rotary_dims) / 2;
    g  = Grid{half, uint32_t(n_heads), uint32_t(N)};
    tg = Threadgroup{half, 1, 1};
}

}  // namespace pie::kernels::rope
