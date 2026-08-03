// Expert-major reordering for a routed FFN.
//
// A mixture's problem at prefill is not arithmetic, it is that every (token,
// slot) pair reads a DIFFERENT expert's weight matrix. The matvec form pays
// that literally: `rows * experts_per_token` matvecs, each re-reading a whole
// [N, K] stack slice, so a 512-token prefill reads the same 128 experts four
// thousand times. Nothing about the shape improves with the batch, which is
// exactly what the measurements said -- our routed prefill was flat in length
// where mlx-lm's climbed.
//
// The fix is not a wider kernel. It is to put the rows that share an expert
// NEXT TO EACH OTHER, at which point the projection is an ordinary batched
// matmul over a contiguous block of rows against one weight slice -- the
// `affine_qmm_t` this driver already has. These three kernels are that
// reordering and nothing else:
//
//   sort    -- a counting sort of the (row, slot) pairs by expert id, laid out
//              so each expert's run starts on a tile boundary.
//   gather  -- copy each sorted position's source row into that order.
//   scatter -- put the results back where the combine step expects them.
//
// The same three run at M=1. A decode sorts eight pairs, gathers eight rows and
// scatters eight back, which is microseconds -- and it means the routed
// dataflow has ONE shape rather than a decode shape and a prefill shape that
// have to be kept agreeing. The batched and unbatched paths differ in exactly
// one number, `tile_rows`: 1 leaves the sort a pure grouping with no padding
// and the projections stay matvecs; 16 rounds every expert's run up to a tile
// and they become matmuls.

#include <metal_stdlib>

using namespace metal;

// Mirrors `shared_kernels::MoeRouteParams`.
struct MoeRouteParams {
    // Live (row, slot) pairs: `rows * experts_per_token`.
    uint n;
    uint n_experts;
    uint experts_per_token;
    // The row granularity each expert's run is padded to. 1 for the matvec
    // path, the matmul's BM for the batched one.
    uint tile_rows;
    // Capacity of `perm` and `row_expert`, in rows. The host sizes this at the
    // worst case -- `n + min(n, n_experts) * (tile_rows - 1)` -- so the sort
    // can never need a bound it was not given.
    uint padded;
    // Row width, for the gather and the scatter.
    uint width;
};

// One lane per expert during the prefix scan, so this is the widest expert
// count this shape serves. `shared_kernels::kRouterMaxExperts` is the same
// number and the geometry refuses anything above it.
constant constexpr uint kMaxExperts = 1024;

/// Group the (row, slot) pairs by expert.
///
/// A single threadgroup: the scan is over `n_experts` (tens to hundreds), and
/// the scatter over `n` (thousands at most), so the parallelism that matters is
/// in the matmul this feeds, not here. Splitting it would need a global
/// histogram and a second dispatch to scan it, which is more synchronisation
/// than the work is worth.
///
/// Outputs, all indexed by SORTED position:
///   perm[p]        the (row, slot) pair at p, or -1 for a padding row
///   row_expert[p]  the expert p reads, for the matvec path
///   tile_expert[t] the expert tile t reads, or -1 for a tile past the end
///
/// `perm` is a permutation of `[0, n)` followed by padding, never a truncation:
/// every pair the router chose gets a position, because a pair silently dropped
/// here is an expert contribution silently zeroed later.
[[kernel]] void moe_route_sort(
    const device int* expert_ids [[buffer(0)]],
    device int* perm            [[buffer(1)]],
    device int* row_expert      [[buffer(2)]],
    device int* tile_expert     [[buffer(3)]],
    constant MoeRouteParams& p  [[buffer(4)]],
    uint lid                    [[thread_position_in_threadgroup]],
    uint nthreads               [[threads_per_threadgroup]]) {
    threadgroup atomic_uint counts[kMaxExperts];
    threadgroup uint base[kMaxExperts];

    const uint E = min(p.n_experts, kMaxExperts);
    const uint tile = p.tile_rows < 1u ? 1u : p.tile_rows;
    const uint tiles = p.padded / tile;

    // Clear first, and clear EVERYTHING: the padding rows are read by the
    // gather and the spare tiles by the matmul, so a stale -1 that was never
    // written is a row of some previous layer's routing.
    for (uint e = lid; e < E; e += nthreads) atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
    for (uint i = lid; i < p.padded; i += nthreads) {
        perm[i] = -1;
        row_expert[i] = 0;
    }
    for (uint t = lid; t < tiles; t += nthreads) tile_expert[t] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint i = lid; i < p.n; i += nthreads) {
        const int e = expert_ids[i];
        if (e >= 0 && uint(e) < E) {
            atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The scan is serial in one lane. `E` is 128 on Qwen3-MoE and the loop body
    // is an add and a store, so this is a few hundred nanoseconds against the
    // milliseconds of matmul it makes possible.
    if (lid == 0) {
        uint at = 0;
        for (uint e = 0; e < E; ++e) {
            const uint c = atomic_load_explicit(&counts[e], memory_order_relaxed);
            base[e] = at;
            if (c > 0) {
                const uint span = ((c + tile - 1u) / tile) * tile;
                for (uint t = at / tile; t < (at + span) / tile && t < tiles; ++t) {
                    tile_expert[t] = int(e);
                }
                at += span;
            }
            // Reused as the per-expert write cursor by the scatter below.
            atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint i = lid; i < p.n; i += nthreads) {
        const int e = expert_ids[i];
        if (e < 0 || uint(e) >= E) continue;
        const uint at = base[e] + atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        if (at < p.padded) {
            perm[at] = int(i);
            row_expert[at] = e;
        }
    }
}

/// Copy each sorted position's source row into sorted order.
///
/// `perm[p]` is a (row, slot) pair, so the row it reads is `perm[p] /
/// experts_per_token` -- the k slots of one token share one activation, which
/// is why this is a broadcast rather than a permutation of equal sizes.
///
/// Padding rows are zeroed rather than left alone. They are multiplied by a
/// real weight and the result is discarded, so their VALUE never matters -- but
/// an unwritten row is whatever the pool held, and bf16 garbage can be inf,
/// which a later `simd_sum` would spread across a tile that does matter.
[[kernel]] void moe_route_gather(
    const device bfloat* x     [[buffer(0)]],
    device bfloat* out         [[buffer(1)]],
    const device int* perm     [[buffer(2)]],
    constant MoeRouteParams& p [[buffer(3)]],
    uint2 gid                  [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.padded) return;
    const int sel = perm[gid.y];
    const uint k = p.experts_per_token < 1u ? 1u : p.experts_per_token;
    out[uint(gid.y) * p.width + gid.x] =
        sel < 0 ? bfloat(0) : x[(uint(sel) / k) * p.width + gid.x];
}

/// Undo the sort, into the [rows, k, width] stack the combine step reads.
///
/// Every live sorted position has a distinct `perm[p]` in `[0, n)`, so this
/// writes each destination row exactly once and leaves none unwritten -- which
/// is the reason the sort must be a permutation and not a filter.
[[kernel]] void moe_route_scatter(
    const device bfloat* in    [[buffer(0)]],
    device bfloat* out         [[buffer(1)]],
    const device int* perm     [[buffer(2)]],
    constant MoeRouteParams& p [[buffer(3)]],
    uint2 gid                  [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.padded) return;
    const int sel = perm[gid.y];
    if (sel < 0) return;
    out[uint(sel) * p.width + gid.x] = in[uint(gid.y) * p.width + gid.x];
}
