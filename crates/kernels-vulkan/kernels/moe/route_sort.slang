#include "common/bf16.slang"

#define ROUTER_MAX_EXPERTS 1024u
#define PIE_SORT_LANES 1024u

#if defined(PIE_ROUTE_SORT)
[[vk::binding(0)]] StructuredBuffer<int> expert_ids;
[[vk::binding(1)]] RWStructuredBuffer<int> perm;
[[vk::binding(2)]] RWStructuredBuffer<int> row_expert;
[[vk::binding(3)]] RWStructuredBuffer<int> tile_expert;
[[vk::binding(4)]] RWStructuredBuffer<int> inv;

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
void main(uint3 local : SV_GroupThreadID) {
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

[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint c = thread.x;
    uint r = thread.y;
    if (c >= pc.width || r >= pc.padded) return;
    int sel = perm[r];
    uint k = pc.experts_per_token < 1u ? 1u : pc.experts_per_token;
    uint pitch = pc.x_pitch != 0u ? pc.x_pitch : pc.width;
    out_[r * pc.width + c] = sel < 0 ? PIE_STORE(0.0) : x[(uint(sel) / k) * pitch + c];
}

#else
PIE_BUFFER_RO(0, sorted);
PIE_BUFFER_RW(1, y);
[[vk::binding(2)]] StructuredBuffer<int> inv;

struct Push {
    uint n;
    uint width;
    uint zero;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint c = thread.x;
    uint r = thread.y;
    if (c >= pc.width || r >= pc.n) return;
    int at = inv[r];
    y[r * pc.width + c] = at < 0 ? PIE_STORE(0.0) : sorted[uint(at) * pc.width + c];
}
#endif

// pie:instantiate route_sort PIE_ROUTE_SORT=1
// pie:instantiate route_gather PIE_ROUTE_GATHER=1
// pie:instantiate route_scatter PIE_ROUTE_SCATTER=1
