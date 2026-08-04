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
//
// There is no third kernel putting the results back. The sort emits the INVERSE
// permutation alongside the forward one -- it knows both, at no cost, at the
// moment it places a pair -- and the combine step reads its k slots through it.
// Undoing a permutation to feed a kernel that is about to gather anyway is a
// dispatch and a full-width buffer spent to make an index arithmetic look
// simpler.
//
// Both run at M=1. A decode sorts eight pairs and gathers eight rows, which is
// microseconds -- and it means the routed
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
/// One threadgroup is not the same as one LANE, though, and the difference was
/// measured: this scan was serial in lane 0 and cost 20 microseconds a layer,
/// which at 48 layers was most of what the reordering took off decode. There is
/// one thread per expert, so the prefix over the experts is a two-level simd
/// scan and each expert writes its own tiles.
///
/// Outputs, all indexed by SORTED position:
///   perm[p]        the (row, slot) pair at p, or -1 for a padding row
///   row_expert[p]  the expert p reads, for the matvec path
///   tile_expert[t] the expert tile t reads, or -1 for a tile past the end
///
/// and, indexed by PAIR rather than by position, the inverse:
///
///   inv[i]         the sorted position of pair i, or -1 if it has no expert
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
    device int* inv             [[buffer(5)]],
    uint lid                    [[thread_position_in_threadgroup]],
    uint nthreads               [[threads_per_threadgroup]]) {
    threadgroup atomic_uint counts[kMaxExperts];
    threadgroup uint base[kMaxExperts];
    threadgroup uint sg_sum[32];

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
    for (uint i = lid; i < p.n; i += nthreads) inv[i] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint i = lid; i < p.n; i += nthreads) {
        const int e = expert_ids[i];
        if (e >= 0 && uint(e) < E) {
            atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each expert's run, rounded up to a whole tile. An expert nothing routed
    // to takes no space at all -- the padding is per TOUCHED expert, which is
    // what keeps the waste bounded when 128 experts see 8 pairs.
    //
    // `simd_prefix_exclusive_sum` has to be reached by every lane of the
    // simdgroup, so the span is computed as zero past `E` rather than branched
    // around.
    const uint span = lid < E
        ? (atomic_load_explicit(&counts[lid], memory_order_relaxed) > 0u
               ? ((atomic_load_explicit(&counts[lid], memory_order_relaxed) + tile - 1u) / tile) * tile
               : 0u)
        : 0u;
    const uint within = simd_prefix_exclusive_sum(span);
    const uint sg = lid / 32u;
    const uint n_sg = (nthreads + 31u) / 32u;
    // The group total from lane 0 rather than from its last lane. `simd_sum` is
    // uniform over the whole simdgroup either way, and lane 0 always exists --
    // where "the last lane" needs a second clause for a partial group that this
    // dispatch shape can never produce, and so could never be tested.
    const uint sg_total = simd_sum(span);  // uniform: every lane must reach it
    if (lid % 32u == 0u) sg_sum[sg] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // At most 32 simdgroups (1024 threads), so this residual scan is 32 adds in
    // one lane rather than 1024.
    if (lid == 0) {
        uint at = 0;
        for (uint i = 0; i < n_sg && i < 32u; ++i) {
            const uint t = sg_sum[i];
            sg_sum[i] = at;
            at += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < E) {
        const uint at = sg_sum[sg] + within;
        base[lid] = at;
        for (uint t = at / tile; t < (at + span) / tile && t < tiles; ++t) {
            tile_expert[t] = int(lid);
        }
        // Reused as the per-expert write cursor by the scatter below.
        atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
    }
    // `tile_expert` is device memory but nothing in THIS kernel reads it, and
    // the device clears above were already published by the first barrier, so
    // what has to be visible here is `base` and the reset cursors.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < p.n; i += nthreads) {
        const int e = expert_ids[i];
        if (e < 0 || uint(e) >= E) continue;
        const uint at = base[e] + atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        if (at < p.padded) {
            perm[at] = int(i);
            row_expert[at] = e;
            inv[i] = int(at);
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

/// Sum a token's k expert outputs, weighted by the router's softmax, reading
/// them where the SORT left them.
///
/// The same arithmetic as `expert_combine`, and deliberately a separate kernel
/// rather than that one taught an optional index: gpt-oss does not sort, so
/// giving it a buffer it must bind and never reads would be the bias slot
/// problem again.
///
/// A slot whose pair never got a position contributes zero. That cannot happen
/// for a routing the geometry accepted -- every id is in range and every pair
/// is placed -- but reading `y` at -1 if it ever did would be a wild load, and
/// the whole reason the sort is a permutation rather than a filter is that a
/// silently dropped expert is a silently wrong answer.
struct ExpertCombineParams {
    uint width;
    uint experts_per_token;
};

[[kernel]] void moe_combine_sorted(
    const device bfloat* y              [[buffer(0)]],
    const device bfloat* expert_weights [[buffer(1)]],
    device bfloat* out                  [[buffer(2)]],
    constant ExpertCombineParams& p     [[buffer(3)]],
    const device int* inv               [[buffer(4)]],
    uint2 gid                           [[thread_position_in_grid]]) {
  const uint c = gid.x;
  if (c >= p.width) return;
  const uint row = gid.y;
  const uint k = p.experts_per_token;
  float acc = 0;
  for (uint e = 0; e < k; ++e) {
    const int at = inv[row * k + e];
    if (at < 0) continue;
    acc += float(expert_weights[row * k + e]) * float(y[uint(at) * p.width + c]);
  }
  out[row * p.width + c] = static_cast<bfloat>(acc);
}

// ── The shared expert ────────────────────────────────────────────────────────
//
// Every routed member of this family -- Qwen3-Next-80B, Qwen3.5-35B-A3B,
// Qwen3.5-122B-A10B -- runs one DENSE FFN beside the routed bank on every
// token, and adds it to the mixture's output under a learned gate:
//
//   y = routed + sigmoid(shared_expert_gate(x)) * shared_expert(x)
//
// The FFN half needs nothing new -- it is the same three projections and the
// same SwiGLU the dense members of this family already run, at
// `shared_expert_intermediate_size`. Only this last line is new, and it is new
// for one reason: the gate is ONE number per token, broadcast across the whole
// hidden row. `attn_gate` looks like it would serve and does not; its gate is
// full width, so it would read `hidden` gate values where there is one.
//
// Fused rather than a multiply and an add, because the alternative writes the
// scaled shared output to a full-width scratch buffer that the very next
// dispatch consumes and nothing else ever reads.
//
// The sigmoid is computed in float from a bf16 logit. That matters at the
// tails: bf16 has eight mantissa bits, so rounding the logit BEFORE the
// nonlinearity moves the gate by up to a few parts in a thousand, on a term
// that is added to every token's residual in every routed layer.
[[kernel]] void shared_expert_combine(
    const device bfloat* routed [[buffer(0)]],   // [rows, width]
    const device bfloat* shared [[buffer(1)]],   // [rows, width]
    const device bfloat* gate   [[buffer(2)]],   // [rows, 1]
    device bfloat* out          [[buffer(3)]],   // [rows, width] (may alias routed)
    constant uint& width        [[buffer(4)]],
    uint2 gid                   [[thread_position_in_grid]]) {
  const uint c = gid.x;
  if (c >= width) return;
  const uint row = gid.y;
  const float g = 1.0f / (1.0f + metal::exp(-float(gate[row])));
  const uint at = row * width + c;
  out[at] = static_cast<bfloat>(float(routed[at]) + g * float(shared[at]));
}
