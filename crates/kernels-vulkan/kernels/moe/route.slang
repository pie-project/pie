// Generic MoE routing: select experts, group rows by expert, gather rows, and
// combine the sorted results. Ported from Metal's route.metal; the ABI moved
// scalar references into params buffers or push constants exactly as the Vulkan
// table states, but the important axes did not: top-k owns a row grid axis and
// sort is one workgroup for the whole routing.
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
#include "moe/params.slang"

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
PIE_PARAMS(3, RouterParams, p);
PIE_BUFFER_RO(4, per_expert_scale);
groupshared float s_logits[ROUTER_MAX_EXPERTS];
groupshared float chosen[ROUTER_MAX_TOPK];
groupshared uint chosen_i[ROUTER_MAX_TOPK];
groupshared float all_max_s;
groupshared float all_sum_s;

[numthreads(1024, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint lid = local.x;
    uint row = group.y;
    uint n = min(p.n_experts, ROUTER_MAX_EXPERTS);
    uint k = min(p.experts_per_token, ROUTER_MAX_TOPK);
    uint pitch = p.logits_pitch != 0u ? p.logits_pitch : n;
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
        if (p.softmax_over_all == 0u) {
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
PIE_PARAMS(4, MoeRouteParams, p);
[[vk::binding(5)]] RWStructuredBuffer<int> inv;
groupshared uint counts[ROUTER_MAX_EXPERTS];
groupshared uint base[ROUTER_MAX_EXPERTS];

[numthreads(1024, 1, 1)]
void main(uint3 local: SV_GroupThreadID) {
    uint lid = local.x;
    uint nthreads = PIE_SORT_LANES;
    uint E = min(p.n_experts, ROUTER_MAX_EXPERTS);
    uint tile = p.tile_rows < 1u ? 1u : p.tile_rows;
    uint tiles = p.padded / tile;

    for (uint e = lid; e < E; e += nthreads) counts[e] = 0u;
    for (uint i = lid; i < p.padded; i += nthreads) { perm[i] = -1; row_expert[i] = 0; }
    for (uint t = lid; t < tiles; t += nthreads) tile_expert[t] = -1;
    for (uint i = lid; i < p.n; i += nthreads) inv[i] = -1;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = lid; i < p.n; i += nthreads) {
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

    for (uint i = lid; i < p.n; i += nthreads) {
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
        if (at < p.padded) {
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
PIE_PARAMS(3, MoeRouteParams, p);

[numthreads(16, 16, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint r = thread.y;
    if (c >= p.width || r >= p.padded) return;
    int sel = perm[r];
    uint k = p.experts_per_token < 1u ? 1u : p.experts_per_token;
    uint pitch = p.x_pitch != 0u ? p.x_pitch : p.width;
    out_[r * p.width + c] = sel < 0 ? PIE_STORE(0.0) : x[(uint(sel) / k) * pitch + c];
}
#elif defined(PIE_COMBINE_SORTED)
PIE_BUFFER_RO(0, y);
PIE_BUFFER_RO(1, expert_weights);
PIE_BUFFER_RW(2, out_);
PIE_PARAMS(3, ExpertCombineParams, p);
[[vk::binding(4)]] StructuredBuffer<int> inv;

[numthreads(16, 16, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint row = thread.y;
    if (c >= p.width) return;
    float acc = 0.0;
    for (uint e = 0u; e < p.experts_per_token; e++) {
        int at = inv[row * p.experts_per_token + e];
        if (at >= 0) acc += PIE_LOAD(expert_weights[row * p.experts_per_token + e]) * PIE_LOAD(y[uint(at) * p.width + c]);
    }
    uint pitch = p.out_pitch != 0u ? p.out_pitch : p.width;
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
// pie:instantiate route_gather PIE_ROUTE_GATHER=1
// pie:instantiate route_sort PIE_ROUTE_SORT=1
// pie:instantiate router_topk_bfloat16 PIE_ROUTER_TOPK=1
// pie:instantiate router_topk_scaled_bfloat16 PIE_ROUTER_TOPK=1 PIE_SCALED=1
// pie:instantiate shared_expert_combine
// pie:instantiate shared_expert_combine_strided PIE_STRIDED=1
