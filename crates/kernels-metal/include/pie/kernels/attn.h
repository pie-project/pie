#pragma once

/// Attention and the KV write: launch shapes.
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

namespace pie::kernels::attn {

using std::uint32_t;

// sdpa_vector_decode: tid.x = q_batch_head_idx (threadgroup-x), one threadgroup
// (1024 threads) per query head → grid=(n_q_heads*1024, 1, 1), tg=(1024,1,1).
// THREE names for this, before it came down here: `sdpa_dispatch` (qwen3.5),
// `sdpa_sliding_dispatch` (gemma4) and `sdpa_sink_dispatch` (gpt-oss). The last
// two had byte-identical bodies and the first is their `rows == 1` case, so the
// `rows` parameter is all that separated them -- three families writing one
// shape three times is exactly what a per-model home for kernel knowledge
// produces, and is why this file exists.
//
// The grid is in THREADS -- `StepEncoder::dispatch` calls `dispatchThreads` --
// so the head count multiplies the threadgroup width rather than standing
// alone. Writing it the other way launches `n_q_heads` threads TOTAL, which is
// not an error the hardware reports: the kernel's simd reductions just read
// lanes that were never dispatched.
inline void sdpa_dispatch(int n_q_heads, Grid& g, Threadgroup& tg, int rows = 1) {
    g  = Grid{uint32_t(n_q_heads) * 1024, uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{1024, 1, 1};
}

// kv_append: tid=(head_dim, n_kv_heads) elementwise scatter to the page.
inline void kv_append_dispatch(int head_dim, int n_kv_heads, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(head_dim), uint32_t(n_kv_heads), 1};
    tg = Threadgroup{uint32_t(head_dim), 1, 1};
}

// q_gate_split: deinterleave qg -> Q + gate. one thread per (channel, query head).
inline void q_split_dispatch(int head_dim, int n_q, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(head_dim), uint32_t(n_q), 1};
    tg = Threadgroup{uint32_t(head_dim), 1, 1};
}

// attn_gate: attn *= sigmoid(gate), elementwise over n_q*head_dim (head-major).
inline void attn_gate_dispatch(int n_q, int head_dim, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(n_q) * uint32_t(head_dim), 1, 1};
    tg = Threadgroup{256, 1, 1};
}


// ── the multi-batch shapes ──────────────────────────────────────────────────
//
// Moved verbatim from `driver-metal`'s `decode_dispatch_mb.hpp`. Every one is a
// pure function of the geometry: the DECISIONS that file also held --
// `qmm_bn`, `qmm_bn_unsplit`, `sdpa_should_tile` -- read `DeviceTuning` and
// stayed behind, so anything they choose arrives here as an argument.

// Query rows per threadgroup in `sdpa_paged_tiled` -- one per simdgroup, and a
// threadgroup is 1024 threads. It is the factor by which that kernel divides
// the K/V traffic, and it must equal the kernel's own QT.
inline constexpr int kSdpaQueryTile = 32;

// The head widths `sdpa_paged_mma.metal` is instantiated for. The matrix path
// stages three tiles of KT*D halves in 32 KB of threadgroup memory, which is
// what bounds the list: adding a width means choosing its KT there first.
inline constexpr int kSdpaMmaHeadDim = 64;

// sdpa_paged_decode: one threadgroup per (q_head, query row). grid=(n_q_heads*1024, N, 1),
// tg=(1024,1,1). Causal bound per row = position_ids[row]; request = req_of_token[row].
inline void sdpa_paged_dispatch(int n_q_heads, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(n_q_heads) * 1024u, uint32_t(N), 1};
    tg = Threadgroup{1024, 1, 1};
}

// sdpa_paged_tiled: one threadgroup per (q_head, tile of kSdpaQueryTile rows).
// grid=(n_q_heads*1024, ceil(N/QT), 1), tg=(1024,1,1). The grid rounds UP, so
// the kernel reads N from bind::SdpaPaged::Rows to retire its partial tile.
inline void sdpa_paged_tiled_dispatch(int n_q_heads, int N, Grid& g, Threadgroup& tg) {
    const uint32_t tiles = uint32_t((N + kSdpaQueryTile - 1) / kSdpaQueryTile);
    g  = Grid{uint32_t(n_q_heads) * 1024u, tiles < 1u ? 1u : tiles, 1};
    tg = Threadgroup{1024, 1, 1};
}

// sdpa_paged_mma: the same tile of `kSdpaQueryTile` rows, but a simdgroup owns
// EIGHT of them and multiplies 8x8 fragments, so the threadgroup is 128 threads
// rather than 1024. Same grid otherwise -- the tile height is what the grid
// describes, and that has not moved.
inline void sdpa_paged_mma_dispatch(int n_q_heads, int N, Grid& g, Threadgroup& tg) {
    const uint32_t tiles = uint32_t((N + kSdpaQueryTile - 1) / kSdpaQueryTile);
    g  = Grid{uint32_t(n_q_heads) * 128u, tiles < 1u ? 1u : tiles, 1};
    tg = Threadgroup{128, 1, 1};
}

// kv_append (paged, delta's kernel): one thread per (channel, kv_head, token). grid=
// (head_dim, n_kv_heads, N). Token m scatters to its phys_slot(position_ids[m]).
inline void kv_append_mb_dispatch(int head_dim, int n_kv_heads, int N, Grid& g, Threadgroup& tg) {
    g  = Grid{uint32_t(head_dim), uint32_t(n_kv_heads), uint32_t(N)};
    tg = Threadgroup{uint32_t(head_dim), 1, 1};
}

}  // namespace pie::kernels::attn
