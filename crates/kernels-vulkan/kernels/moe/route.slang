// Generic MoE routing: select experts, group rows by expert, gather rows, and
// combine the sorted results. Ported from Metal's route.metal; the important
// axes survived the port unchanged -- top-k owns a row grid axis and sort is one
// workgroup for the whole routing.
//
// EVERY SCALAR HERE IS A PUSH CONSTANT NOW, AND NONE IS A STORAGE STRUCT.
// `RouterParams`, `MoeRouteParams` and `ExpertCombineParams` came over from
// `route.metal` whole, as `PIE_PARAMS` blocks on `@binding(3)`, `@binding(4)`
// and `@binding(3)`, and a storage block costs this plane three things a push
// range does not: a descriptor in every set the module declares, a slot in the
// staging buffer `serve::fire` gathers after every rectangle is planned, and the
// sentinel `Holds::params_block` has to mint because at the moment an arm runs
// the memory that block will live in does not exist yet. The routines state
// `Const<u32>` marks in their place -- the same words of the same statement run,
// reached by index rather than by offset -- and the widest of the four blocks is
// twenty-eight bytes, which is a push range on every device this targets
// (`Device::max_push`; the guaranteed floor is 128).
//
// THE FIELD ORDER IS THE STRUCT'S, BECAUSE IT IS THE STATEMENT'S. `words` packs
// the scalars a body passed in the order it passed them and `Device::dispatch`
// refuses a run whose length is not exactly the reflected range, so a member
// added or dropped here is loud. A member REORDERED is not: it is the same
// number of bytes with two of the numbers swapped, and a router that read
// `experts_per_token` as its expert count would route, and weight, and be wrong.
//
// DELETING THE BLOCKS RENUMBERED THE BINDINGS AFTER THEM. Three of the four sat
// in the MIDDLE of their arm's operand list -- the sort's at 4 of 6 with `inv`
// after it, the combine's at 3 of 5 with `inv` after it, the router's at 3 of 5
// with the per-expert scale after it -- and `encode`'s descriptor set is written
// densely from the buffers the body passed. Each moved operand says so where it
// is declared.
//
// The Slang port changed spellings only, and three of them are worth naming.
// The two `#extension` lines are gone -- Slang derives its capabilities from
// the intrinsics a module actually uses. `atomicAdd` is `InterlockedAdd`, whose
// previous value is an OUT PARAMETER rather than the result, which is the one
// rewrite in this file that a careless hand would get backwards: `route_sort`'s
// second pass reads that previous value as the row's slot within its expert.
// And `gl_WorkGroupSize.x` has no Slang spelling, so the sort's walk stride is
// `PIE_SORT_LANES`, stated once beside `[numthreads]`.

#include "common/bf16.slang"

#define ROUTER_MAX_TOPK 16u
#define ROUTER_MAX_EXPERTS 1024u

/// The sort's workgroup, and therefore every walk stride in it. One workgroup
/// covers the whole routing, so a stride that did not match the declaration
/// would leave rows unvisited -- and an unvisited row keeps `perm = -1`, which
/// `route_gather` reads as a zero row rather than as an error.
#define PIE_SORT_LANES 1024u

#if defined(PIE_ROUTER_TOPK)
PIE_BUFFER_RO(0, logits);
[[vk::binding(1)]] RWStructuredBuffer<int> expert_ids;
PIE_BUFFER_RW(2, expert_weights);
// MOVED DOWN A BINDING: `RouterParams` was 3 and this was 4. The binding stays
// declared outside the `PIE_SCALED` guard so the descriptor exists in both
// modules -- slangc deletes it from the unscaled one for being unread, which is
// the whole reason the unscaled routine can drop the operand rather than bind a
// slot it never dereferences.
PIE_BUFFER_RO(3, per_expert_scale);

// `RouterParams`, field for field and in its order. Sixteen bytes.
//
// `logits_pitch` of zero means the pitch IS `n_experts`, which is load-bearing:
// a router reading a slice of a wider activation has a pitch that is not its
// expert count, and a host with no slice writes 0 rather than restating the
// count. `softmax_over_all` decides the DENOMINATOR of every routing weight,
// and is the word `driver-metal/tests/packed_params_cover_the_struct.rs` was
// written about -- a text that stated two words left this one holding the next
// dispatch's first scalar, and both readings produce weights.
struct Push {
    uint n_experts;
    uint experts_per_token;
    uint softmax_over_all;
    uint logits_pitch;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
groupshared float s_logits[ROUTER_MAX_EXPERTS];
groupshared float chosen[ROUTER_MAX_TOPK];
groupshared uint chosen_i[ROUTER_MAX_TOPK];
groupshared float all_max_s;
groupshared float all_sum_s;

[numthreads(1024, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint lid = local.x;
    uint row = group.y;
    uint n = min(pc.n_experts, ROUTER_MAX_EXPERTS);
    uint k = min(pc.experts_per_token, ROUTER_MAX_TOPK);
    uint pitch = pc.logits_pitch != 0u ? pc.logits_pitch : n;
    float neg = -3.0e38;

    if (lid < ROUTER_MAX_EXPERTS) s_logits[lid] = lid < n ? PIE_LOAD(logits[row * pitch + lid]) : neg;
    GroupMemoryBarrierWithGroupSync();

    if (lid == 0u) {
        float all_max = neg;
        for (uint e = 0u; e < n; e++) all_max = max(all_max, s_logits[e]);
        float all_sum = 0.0;
        for (uint e = 0u; e < n; e++) all_sum += exp(s_logits[e] - all_max);
        all_max_s = all_max;
        all_sum_s = all_sum;

        for (uint r = 0u; r < k; r++) {
            float best = neg;
            uint best_i = 0xffffffffu;
            for (uint e = 0u; e < n; e++) {
                float v = s_logits[e];
                if (v > best) { best = v; best_i = e; }
            }
            chosen[r] = best;
            chosen_i[r] = best_i;
            if (best_i < n) s_logits[best_i] = neg;
        }

        float mx = all_max_s;
        float sum = all_sum_s;
        if (pc.softmax_over_all == 0u) {
            mx = neg;
            for (uint r = 0u; r < k; r++) mx = max(mx, chosen[r]);
            sum = 0.0;
            for (uint r = 0u; r < k; r++) sum += exp(chosen[r] - mx);
        }
        for (uint r = 0u; r < k; r++) {
            uint e = chosen_i[r];
            float w = exp(chosen[r] - mx) / sum;
#if defined(PIE_SCALED)
            w *= PIE_LOAD(per_expert_scale[e]);
#endif
            expert_ids[row * k + r] = int(e);
            expert_weights[row * k + r] = PIE_STORE(w);
        }
    }
}
#elif defined(PIE_ROUTE_SORT)
[[vk::binding(0)]] StructuredBuffer<int> expert_ids;
[[vk::binding(1)]] RWStructuredBuffer<int> perm;
[[vk::binding(2)]] RWStructuredBuffer<int> row_expert;
[[vk::binding(3)]] RWStructuredBuffer<int> tile_expert;
// MOVED DOWN A BINDING: `MoeRouteParams` was 4 and this was 5.
[[vk::binding(4)]] RWStructuredBuffer<int> inv;

// `MoeRouteParams`, field for field, and SHARED WITH `route_gather` below.
// Twenty-eight bytes, the widest push block in this file.
//
// One layout for the sort and the gather so that the padding one writes and the
// bounds the other reads cannot disagree. That is not two structs that happen to
// look alike: `model-dsl` states the same seven words for both statements, and
// both routines take the same seven marks in the same order -- three of which
// the gather never reads and carries anyway, so that `width` is word 5 and
// `x_pitch` word 6 in both.
//
// `n` is the number of (row, slot) PAIRS -- one per expert choice -- while
// `padded` is the length of the permutation, `n` rounded up so every expert's
// span is a whole number of `tile_rows` tiles. The two are different numbers and
// this kernel reads both; a body that used one for the other would clear a
// permutation shorter than it fills.
struct Push {
    uint n;
    uint n_experts;
    uint experts_per_token;
    uint tile_rows;
    uint padded;
    uint width;
    uint x_pitch;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
groupshared uint counts[ROUTER_MAX_EXPERTS];
groupshared uint base[ROUTER_MAX_EXPERTS];

[numthreads(1024, 1, 1)]
void main(uint3 local: SV_GroupThreadID) {
    uint lid = local.x;
    uint nthreads = PIE_SORT_LANES;
    uint E = min(pc.n_experts, ROUTER_MAX_EXPERTS);
    uint tile = pc.tile_rows < 1u ? 1u : pc.tile_rows;
    uint tiles = pc.padded / tile;

    for (uint e = lid; e < E; e += nthreads) counts[e] = 0u;
    for (uint i = lid; i < pc.padded; i += nthreads) { perm[i] = -1; row_expert[i] = 0; }
    for (uint t = lid; t < tiles; t += nthreads) tile_expert[t] = -1;
    for (uint i = lid; i < pc.n; i += nthreads) inv[i] = -1;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = lid; i < pc.n; i += nthreads) {
        int e = expert_ids[i];
        if (e >= 0 && uint(e) < E) {
            uint previous;
            InterlockedAdd(counts[uint(e)], 1u, previous);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (lid == 0u) {
        uint at = 0u;
        for (uint e = 0u; e < E; e++) {
            uint c = counts[e];
            uint span = c > 0u ? ((c + tile - 1u) / tile) * tile : 0u;
            base[e] = at;
            for (uint t = at / tile; t < (at + span) / tile && t < tiles; t++) tile_expert[t] = int(e);
            counts[e] = 0u;
            at += span;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = lid; i < pc.n; i += nthreads) {
        int e = expert_ids[i];
        if (e < 0 || uint(e) >= E) continue;
        // The PREVIOUS value is the row's slot within its expert's run, which
        // is why this atomic's result cannot be dropped. `InterlockedAdd`
        // returns nothing and hands it back through an out parameter, so a
        // port that read the call as an expression would have compiled to a
        // counter that advanced and a slot that never did.
        uint previous;
        InterlockedAdd(counts[uint(e)], 1u, previous);
        uint at = base[uint(e)] + previous;
        if (at < pc.padded) {
            perm[at] = int(i);
            row_expert[at] = e;
            inv[i] = int(at);
        }
    }
}
#elif defined(PIE_ROUTE_GATHER)
PIE_BUFFER_RO(0, x);
PIE_BUFFER_RW(1, out_);
[[vk::binding(2)]] StructuredBuffer<int> perm;

// THE SORT'S BLOCK, WHOLE, and three of these seven words are never read here.
//
// Deliberate, and see `route_sort`'s note for why: the sort's `padded` is this
// kernel's row bound, so a gather carrying its own shorter block would be a
// second place for the same number to be stated and a second place for it to be
// wrong. An unread member of a push block is not eliminated -- `quant/qmm_t.slang`
// has carried two of them through `-O2` since the cast entrypoints were split --
// so the reflected range is the full twenty-eight bytes and the routine states
// all seven marks.
struct Push {
    uint n;
    uint n_experts;
    uint experts_per_token;
    uint tile_rows;
    uint padded;
    uint width;
    uint x_pitch;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(16, 16, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint r = thread.y;
    if (c >= pc.width || r >= pc.padded) return;
    int sel = perm[r];
    uint k = pc.experts_per_token < 1u ? 1u : pc.experts_per_token;
    uint pitch = pc.x_pitch != 0u ? pc.x_pitch : pc.width;
    out_[r * pc.width + c] = sel < 0 ? PIE_STORE(0.0) : x[(uint(sel) / k) * pitch + c];
}
#elif defined(PIE_EXPERT_COMBINE)
// Sum a token's k expert outputs, weighted by the router, reading them WHERE
// THEY WERE WRITTEN.
//
// `routed` is `[rows * k, width]` in (token, slot) order — one row per route,
// in the order the routes were chosen — so slot `e` of token `n` is at row
// `n * k + e` and there is no permutation to consult. That is the whole
// difference from `combine_sorted` below, and it is why the two are two arms
// rather than one taught an optional index: every sorted family binds an
// inverse permutation, and a caller whose rows were never moved should not
// carry a buffer it would only fill with the identity.
//
// **THIS PLANE HAD ONLY THE SORTED ONE**, so `moe.weighted_sum` was in the
// claim table and refused every fire — the permutation `combine_sorted` folds
// through is written by `route_sort`, which no point of this plane claims. A
// whole tower is what found it: `driver-vulkan/tests/banked_argmax.rs` walked
// a gpt-oss decode and stopped here, at op 24.
//
// THE WEIGHTS ARE FLOAT. `Moe::weighted_sum` declares them
// `In<Self::Tensor<f32>>`, because a router weight is a probability and the
// fold multiplies in float on every plane; `combine_sorted`'s are `PIE_ACT`
// because that is the element the legacy routed families stage them at.
PIE_BUFFER_RO(0, routed);
[[vk::binding(1)]] StructuredBuffer<float> expert_weights;
PIE_BUFFER_RW(2, out_);

struct Push {
    uint width;
    uint rows;
    uint experts_per_token;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(16, 16, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint row = thread.y;
    // BOTH AXES GUARDED. The grid is `elementwise_rows(width, rows)` and the
    // driver rounds each axis up to the module's own `[numthreads]`, so a
    // width of 2880 and 16 lanes fits exactly while a single decode row does
    // not — fifteen of every sixteen `row` lanes are past the end.
    if (c >= pc.width || row >= pc.rows) return;
    float acc = 0.0;
    for (uint e = 0u; e < pc.experts_per_token; e++) {
        uint at = row * pc.experts_per_token + e;
        acc += expert_weights[at] * PIE_LOAD(routed[at * pc.width + c]);
    }
    out_[row * pc.width + c] = PIE_STORE(acc);
}
#elif defined(PIE_COMBINE_SORTED)
PIE_BUFFER_RO(0, y);
PIE_BUFFER_RO(1, expert_weights);
PIE_BUFFER_RW(2, out_);
// MOVED DOWN A BINDING: `ExpertCombineParams` was 3 and this was 4.
[[vk::binding(3)]] StructuredBuffer<int> inv;

// `ExpertCombineParams`, field for field. Twelve bytes.
//
// `out_pitch` of zero means `width`, for the same reason `logits_pitch`'s zero
// means `n_experts`: the mixture's output lands in whatever layout the caller's
// activations are in -- a batched decode's are packed, a prefill's are a uniform
// scratch stride apart -- and a host with nothing to say writes 0 rather than
// restating the width.
struct Push {
    uint width;
    uint experts_per_token;
    uint out_pitch;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(16, 16, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint row = thread.y;
    if (c >= pc.width) return;
    float acc = 0.0;
    for (uint e = 0u; e < pc.experts_per_token; e++) {
        int at = inv[row * pc.experts_per_token + e];
        if (at >= 0) acc += PIE_LOAD(expert_weights[row * pc.experts_per_token + e]) * PIE_LOAD(y[uint(at) * pc.width + c]);
    }
    uint pitch = pc.out_pitch != 0u ? pc.out_pitch : pc.width;
    out_[row * pitch + c] = PIE_STORE(acc);
}
#else
PIE_BUFFER_RO(0, routed);
PIE_BUFFER_RO(1, shared_);
PIE_BUFFER_RO(2, gate);
PIE_BUFFER_RW(3, out_);
struct Push {
    uint width;
#if defined(PIE_STRIDED)
    int row_pitch;
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(16, 16, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint r = thread.y;
    if (c >= pc.width) return;
    // The data base and the GATE index are two different numbers, and this
    // body used to keep them in one variable. `gate` is one value per row, so
    // the non-strided read is `gate[r]`; folding it into `r * width` -- the
    // base of the row's DATA -- reads `width` elements past the gate for every
    // row after the first, which for a `rows`-long allocation is out of bounds
    // almost immediately and blends every row but row 0 with a garbage weight.
    //
    // The strided variant is the opposite and is NOT a copy of this: there the
    // gate really does stride by the pitch, because `qmv_out_size` answers 1
    // for the shared gate projection and so its output is written a full pitch
    // apart like every other projection's. `route.metal` states both halves,
    // and the port collapsed the one where they differ.
#if defined(PIE_STRIDED)
    uint base = r * uint(pc.row_pitch);
    uint gate_at = base;
#else
    uint base = r * pc.width;
    uint gate_at = r;
#endif
    float g = 1.0 / (1.0 + exp(-PIE_LOAD(gate[gate_at])));
    uint at = base + c;
    out_[at] = PIE_STORE(PIE_LOAD(routed[at]) + g * PIE_LOAD(shared_[at]));
}
#endif

// pie:instantiate combine_sorted PIE_COMBINE_SORTED=1
// pie:instantiate expert_combine PIE_EXPERT_COMBINE=1
// pie:instantiate route_gather PIE_ROUTE_GATHER=1
// pie:instantiate route_sort PIE_ROUTE_SORT=1
// pie:instantiate router_topk_bfloat16 PIE_ROUTER_TOPK=1
// pie:instantiate router_topk_scaled_bfloat16 PIE_ROUTER_TOPK=1 PIE_SCALED=1
// pie:instantiate shared_expert_combine
// pie:instantiate shared_expert_combine_strided PIE_STRIDED=1
