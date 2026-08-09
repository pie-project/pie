#pragma once

/// Gathers and embeddings: launch shapes.
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

namespace pie::kernels::layout {

using std::uint32_t;

// embed_gather_4bit: one thread per output channel (hidden), tg flexible.
inline void embed_dispatch(int hidden, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(hidden), 1, 1};
    tg = Threadgroup{256, 1, 1};
}


// ── the multi-batch shapes ──────────────────────────────────────────────────
//
// Moved verbatim from `driver-metal`'s `decode_dispatch_mb.hpp`. Every one is a
// pure function of the geometry: the DECISIONS that file also held --
// `qmm_bn`, `qmm_bn_unsplit`, `sdpa_should_tile` -- read `DeviceTuning` and
// stayed behind, so anything they choose arrives here as an argument.

// embed_gather_mb over N tokens: thread (channel k, token m). Token m gathers id[m].
// out token-major [N, hidden]. grid=(hidden, N, 1), tg=(256,1,1).
inline void embed_mb_dispatch(int hidden, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(hidden), uint32_t(N), 1};
    tg = Threadgroup{256, 1, 1};
}

}  // namespace pie::kernels::layout
